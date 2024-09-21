import os
import glob
import subprocess

def run_colmap(
        project_path: str, 
        image_dir: str = "images", 
        path_to_colmap: str = "colmap",
        camera_model: str = "OPENCV",
        matcher: str = "exhaustive",
        single_camera: bool = True,
        max_num_features: int = 2048,
        use_gpu: bool = False
        ) -> None:
    print("Estimating camera poses and initial point cloud")

    image_path = os.path.join(project_path, image_dir)
    database_path = os.path.join(project_path, "database.db")

    sparse_path = os.path.join(project_path, 'sparse')
    if not os.path.isdir(sparse_path):
        os.makedirs(sparse_path)

    # Check whether COLMAP was executed on project already
    if os.path.isdir(os.path.join(sparse_path, "0")):
        print(f"Warning: Skipping COLMAP")
        return

    subprocess.call([
        path_to_colmap, "feature_extractor",
        "--image_path", image_path,
        "--database_path", database_path,
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", ("1" if single_camera else "0"),
        "--SiftExtraction.max_num_features", str(max_num_features),
        "--SiftExtraction.use_gpu", ("1" if use_gpu else "0")
    ])

    subprocess.call([
        path_to_colmap, f"{matcher}_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", ("1" if use_gpu else "0")
    ])

    subprocess.call([
        path_to_colmap, "mapper",
        "--image_path", image_path,
        "--database_path", database_path,
        "--output_path", sparse_path
    ])

    for output_path in glob.glob(os.path.join(sparse_path, "*")):             
        subprocess.call([
            path_to_colmap, "model_converter",
            "--input_path", output_path,
            "--output_path", output_path,
            "--output_type", "TXT"
        ])

if __name__ == '__main__':
    run_colmap(
        project_path="datasets/kitchen",
        path_to_colmap="C:/Program Files/Colmap/COLMAP.bat"
        )