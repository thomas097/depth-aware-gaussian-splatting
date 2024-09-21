import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


class DepthAnything:
    def __init__(self, model_size: str = "small") -> None:
        assert model_size in {"small", "base", "large"}
        self._pipe = pipeline(
            task="depth-estimation", 
            model=f"depth-anything/Depth-Anything-V2-{model_size.title()}-hf"
            )

    def from_file(self, filepath: str) -> Image.Image:
        image = Image.open(filepath).convert("RGB")
        return self._pipe(image)["depth"]
    

def run_depth_estimation(project_path: str, image_path: str = 'images', model_size: str = 'small') -> None:
    print("Estimating depth images")
    model = DepthAnything(model_size=model_size)

    depth_dir = os.path.join(project_path, "depth")
    if not os.path.isdir(depth_dir):
        os.makedirs(depth_dir)

    image_dir = os.path.join(project_path, image_path, "*")
    for image_file in tqdm(glob.glob(image_dir)):

        gt_depth = model.from_file(image_file)

        depth_file = os.path.join(project_path, 'depth', os.path.basename(image_file))
        gt_depth.save(depth_file)

    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filepath = "datasets/kitchen/images/frame_00001.jpg"
    outfile = "datasets/kitchen/depth/frame_00001.jpg"

    depth = DepthAnything().from_file(filepath)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(filepath))
    plt.subplot(1, 2, 2)
    plt.imshow(depth)
    plt.show()

    plt.imsave(outfile, depth)