from PIL import Image
import os

def enforce_image_format_and_size(image_path: str, size=(1024, 1024), fmt="PNG"):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    base, _ = os.path.splitext(image_path)
    new_path = base + ".png"
    img.save(new_path, format=fmt)
    if new_path != image_path:
        try:
            os.remove(image_path)
        except:
            pass
    return new_path
