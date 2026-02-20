import random
import string
from datetime import date
from pathlib import Path

from text_utils import generate_random_txt_img

NUM_IMAGES = 10
NUM_CHARACTERS = 29000
IMG_SHAPE = (1600, 900)
TEXT_SIZE = 22


def build_output_dir():
    base_name = date.today().strftime("%b-%d-%Y")
    output_dir = Path(base_name)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return output_dir

    suffix = 2
    while True:
        candidate = Path(f"{base_name}{suffix}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def main():
    save_path = build_output_dir()
    image_name_prefix = "generated_text"

    for i in range(NUM_IMAGES):
        text = "".join(
            random.choices(string.ascii_letters + string.digits, k=NUM_CHARACTERS)
        )

        text_color = random.choices(["black", "white"], weights=(70, 30), k=1)[0]
        background_color = "black" if text_color == "white" else "white"

        generate_random_txt_img(
            text,
            IMG_SHAPE,
            TEXT_SIZE,
            text_color,
            background_color,
            save_path / f"{image_name_prefix}{i}.png",
        )


if __name__ == "__main__":
    main()
