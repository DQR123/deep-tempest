import argparse
import logging
import os

import numpy as np
import torch

from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

try:
    import fastwer
    import pytesseract
    HAS_OCR_DEPS = True
except ImportError:
    HAS_OCR_DEPS = False


"""
Testing code for DRUNet.
Refactored for improved portability and safer execution on Windows/Linux.
"""


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def calculate_cer_wer(img_e, img_h):
    if not HAS_OCR_DEPS:
        return float("nan"), float("nan")

    text_h = pytesseract.image_to_string(img_h).strip().replace("\n", " ")
    text_e = pytesseract.image_to_string(img_e).strip().replace("\n", " ")

    cer = fastwer.score_sent(text_e, text_h, char_level=True)
    wer = fastwer.score_sent(text_e, text_h)
    return cer, wer


def _crop_to_same_shape(img_a, img_b):
    min_h = min(img_a.shape[0], img_b.shape[0])
    min_w = min(img_a.shape[1], img_b.shape[1])
    return img_a[:min_h, :min_w], img_b[:min_h, :min_w]


def main(json_path="options/test_drunet.json"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default=json_path, help="Path to option JSON file.")
    parser.add_argument("--dist", default=False)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)
    opt["dist"] = _to_bool(args.dist)

    logger_name = "test"
    utils_logger.logger_info(logger_name, os.path.join(opt["path"]["log"], f"{logger_name}.log"))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    if opt["dist"]:
        init_dist("pytorch")
    opt["rank"], opt["world_size"] = get_dist_info()
    opt = option.dict_to_nonedict(opt)

    model_path = opt["path"]["pretrained_netG"]
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_epoch = os.path.basename(model_path).split("_G")[0]

    opt_netg = opt["netG"]
    from models.network_unet import UNetRes as net

    model = net(
        in_nc=opt_netg["in_nc"],
        out_nc=opt_netg["out_nc"],
        nc=opt_netg["nc"],
        nb=opt_netg["nb"],
        act_mode=opt_netg["act_mode"],
        bias=opt_netg["bias"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)
    for _, value in model.named_parameters():
        value.requires_grad = False

    logger.info("Model path: %s", model_path)
    logger.info("Params number: %s", sum(x.numel() for x in model.parameters()))

    l_paths = util.get_image_paths(opt["datasets"]["test"]["dataroot_L"])
    h_paths = util.get_image_paths(opt["datasets"]["test"]["dataroot_H"])
    noise_sigma = opt["datasets"]["test"]["sigma_test"]

    if len(l_paths) != len(h_paths):
        logger.warning("Uneven test sets: L=%d, H=%d. Processing paired minimum length.", len(l_paths), len(h_paths))

    metrics = {
        "psnr": 0.0,
        "ssim": 0.0,
        "edge_jaccard": 0.0,
        "cer": 0.0,
        "wer": 0.0,
    }
    processed = 0

    for l_path, h_path in zip(l_paths, h_paths):
        processed += 1
        image_name_ext = os.path.basename(l_path)
        img_name, _ = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt["path"]["images"], img_name)
        util.mkdir(img_dir)

        img_l_original = util.imread_uint(l_path, n_channels=3)[50:-50, 100:-100, :]
        img_l = util.uint2single(img_l_original[:, :, :2])
        img_l = util.single2tensor4(img_l)

        if noise_sigma > 0:
            noise_level = torch.FloatTensor([int(noise_sigma)]) / 255.0
            img_l.add_(torch.randn(img_l.size()).mul_(noise_level).float())
        img_l = img_l.to(device)

        with torch.no_grad():
            img_e = model(img_l)

        img_l_tmp = util.tensor2uint(img_l)
        noisy_l = np.zeros_like(img_l_original)
        noisy_l[:, :, :2] = img_l_tmp
        img_e = util.tensor2uint(img_e)

        util.imsave(noisy_l, os.path.join(img_dir, f"{img_name}_{noise_sigma}std.png"))
        util.imsave(img_e, os.path.join(img_dir, f"{img_name}_model{model_epoch}_{noise_sigma}std.png"))

        img_h = util.imread_uint(h_path, n_channels=3)
        if img_h.ndim == 3:
            img_h = np.mean(img_h, axis=2).astype("uint8")

        img_e, img_h = _crop_to_same_shape(img_e, img_h)

        current_psnr = util.calculate_psnr(img_e, img_h)
        current_ssim = util.calculate_ssim(img_e, img_h)
        current_edge_jaccard = util.calculate_edge_jaccard(img_e, img_h)
        current_cer, current_wer = calculate_cer_wer(img_e, img_h)

        logger.info(
            "%4d--> %10s | PSNR = %.2fdB ; SSIM = %.3f ; edgeJaccard = %.3f ; CER = %.3f%% ; WER = %.3f%%",
            processed,
            image_name_ext,
            current_psnr,
            current_ssim,
            current_edge_jaccard,
            current_cer,
            current_wer,
        )

        metrics["psnr"] += current_psnr
        metrics["ssim"] += current_ssim
        metrics["edge_jaccard"] += current_edge_jaccard
        metrics["cer"] += current_cer
        metrics["wer"] += current_wer

    if processed == 0:
        logger.warning("No test images were found. Verify dataroot_L and dataroot_H in the options file.")
        return

    logger.info(
        "[Average metrics] PSNR : %.2fdB, SSIM = %.3f : edgeJaccard = %.3f : CER = %.3f%% : WER = %.3f%%",
        metrics["psnr"] / processed,
        metrics["ssim"] / processed,
        metrics["edge_jaccard"] / processed,
        metrics["cer"] / processed,
        metrics["wer"] / processed,
    )


if __name__ == "__main__":
    main()
