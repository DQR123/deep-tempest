from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


BLOCKED_FONT_KEYWORDS = {
    "lohit",
    "kacst",
    "navilu",
    "telu",
    "lyx",
    "malayalam",
    "tlwg",
    "samyak",
    "droid",
    "kalapi",
    "openoffice",
    "orya",
}


def _get_readable_system_fonts():
    """Return TTF/OTF fonts that are usually readable across OSes."""
    system_fonts = font_manager.findSystemFonts()
    valid_extensions = {".ttf", ".otf"}
    readable_fonts = []

    for font_path in system_fonts:
        lowercase_path = font_path.lower()
        if Path(lowercase_path).suffix not in valid_extensions:
            continue
        if any(keyword in lowercase_path for keyword in BLOCKED_FONT_KEYWORDS):
            continue
        readable_fonts.append(font_path)

    return readable_fonts


def _load_font(font_path, text_size):
    """Load font and provide cross-platform fallbacks (Linux/Windows/macOS)."""
    if font_path:
        try:
            return ImageFont.truetype(font=font_path, size=text_size)
        except OSError:
            pass

    fallback_candidates = [
        # Windows common fonts
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        # Linux common fonts
        "/usr/share/fonts/truetype/liberation2/LiberationSans-BoldItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for fallback_path in fallback_candidates:
        try:
            return ImageFont.truetype(font=fallback_path, size=text_size)
        except OSError:
            continue

    return ImageFont.load_default()

def generate_random_txt_img(text, img_shape, text_size, text_color, background_color, save_path):
    # Create white plain image
    imagen = Image.new("RGB", img_shape, background_color)
    dibujo = ImageDraw.Draw(imagen)

    # Compute amount of lines depending on image shape and number of characters
    N_total = len(text)
    N_lines = N_total//img_shape[1]
    N_horizontal = int(1.6 * img_shape[0] // (text_size))

    readable_fonts = _get_readable_system_fonts()

    # Write over image one font per line
    for iter in range(N_lines):
        random_font = random.choice(readable_fonts) if readable_fonts else None
        fuente = _load_font(random_font, text_size)

        # Get line text
        texto_linea = text[iter * N_horizontal : (iter+1) * N_horizontal]

        # Adjust text position
        posicion_texto = ((imagen.width - fuente.getsize(texto_linea)[0]) // 2, 
                          int(1.25* iter * text_size)
                          )

        # Write text
        dibujo.text(posicion_texto, texto_linea, font=fuente, fill=text_color)

    # Save image
    imagen.save(save_path)
