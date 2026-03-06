from typing import List
from PIL import Image, ImageDraw


def draw_boxes_on_image(image: Image.Image, boxes: List[List[float]], color: str = "red") -> Image.Image:
    draw = ImageDraw.Draw(image)
    for b in boxes:
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return image
