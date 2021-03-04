import zeus.notebook_utils.syspath as syspath
syspath.add_parent_folder()

import os
import matplotlib.pyplot as plt
import PIL.Image
from zeus.plotting.utils import axes
from kidney.datasets.kaggle import get_reader, SampleType

reader = get_reader()

size = 256
output_dir = "/mnt/fast/data/tiles"
keys = reader.get_keys(SampleType.Unlabeled)

os.makedirs(output_dir, exist_ok=True)

image_shapes = []

for key in keys:
    print("Processing key:", key)
    sample = reader.fetch_one(key)
    image = sample["image"]
    h, w, _ = image.shape

    for y_off in range(0, h - size, size):
        for x_off in range(0, w - size, size):

            x1, y1 = x_off, y_off
            x2, y2 = x_off + size, y_off + size
            name = f"{key}.{x1}.{y1}.{x2}.{y2}.png"
            print(".. creating:", name)

            cut = image[y_off:y_off+size, x_off:x_off+size]
            pil_image = PIL.Image.fromarray(cut)
            pil_image.save(os.path.join(output_dir, name), format="png")

    image_shapes.append({
        "id": key,
        "height": h,
        "width": w,
        "tile_size": size
    })
    
import pandas as pd
df = pd.DataFrame(image_shapes)
df.to_csv("/mnt/fast/data/tiles/image_shapes.csv", index=False)
