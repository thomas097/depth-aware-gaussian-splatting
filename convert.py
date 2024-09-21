from utils.colmap_utils import run_colmap
from utils.depth_estimation_utils import run_depth_estimation

if __name__ == '__main__':
    PROJECT_PATH = "datasets/kitchen"
    PATH_TO_COLMAP = "C:/Program Files/Colmap/COLMAP.bat"
    CAMERA_MODEL = "PINHOLE"
    MODEL_SIZE = "small"

    # Estimate camera parameters and initial pointcloud with COLMAP
    run_colmap(project_path=PROJECT_PATH, camera_model=CAMERA_MODEL, path_to_colmap=PATH_TO_COLMAP)

    # Estimate depth for every image using DepthAnything V2
    run_depth_estimation(project_path=PROJECT_PATH, model_size=MODEL_SIZE)