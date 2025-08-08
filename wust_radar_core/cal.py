import json
import numpy as np
import cv2
from typing import Tuple, Optional
import yaml


class PnPSolver:
    def __init__(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, verbose=False
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.verbose = verbose

    @classmethod
    def from_config(cls, config: dict | str) -> "PnPSolver":
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        camera_matrix = np.array(config["transform"]["K"], dtype=np.float32)
        dist_coeffs = np.array(config["transform"]["dist_coeffs"], dtype=np.float32)
        return cls(camera_matrix, dist_coeffs, config["transform"].get("verbose", True))

    def solve(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            raise ValueError("At least 4 points are required for PnP.")
        if object_points.shape[0] != image_points.shape[0]:
            raise ValueError("Mismatch between 3D and 2D points count.")

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return False, None, None, None

        self.rvec = rvec
        self.tvec = tvec
        R, _ = cv2.Rodrigues(rvec)
        residual = self.calculate_residual(object_points, image_points)

        if self.verbose:
            print("Rotation matrix R:\n", R)
            print("Translation vector t:\n", tvec)
            print("Projection residual (mean):", residual)

        return True, R, tvec, residual

    def calculate_residual(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> float:
        projected_points, _ = cv2.projectPoints(
            objectPoints=object_points,
            rvec=self.rvec,
            tvec=self.tvec,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
        )
        projected_points = projected_points.reshape(-1, 2)
        self.projected_points = projected_points
        self.image_point = image_points
        residual = np.linalg.norm(projected_points - image_points, axis=1)
        return np.mean(residual)

    def draw_visualize_image(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        for pt in self.projected_points:
            x, y = pt.astype(int)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # red
        for pt in self.image_point:
            x, y = pt.astype(int)
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # green
        return img


if __name__ == "__main__":
    # 修改路径为你自己的
    config_path = "config/p.yaml"
    image_path = "output/0.jpg"
    json_path = "output/0.json"
    keypoints_3d_path = "keypoint_6.txt"

    # 加载2D点
    with open(json_path, "r") as f:
        image_points_dict = json.load(f)
    image_points = np.array(
        [image_points_dict[str(i)] for i in range(1, len(image_points_dict) + 1)],
        dtype=np.float32,
    )

    # 加载3D点
    object_points = np.loadtxt(keypoints_3d_path, dtype=np.float32)

    # 可选择性索引6个点
    selected_indices = np.array([0, 1, 2, 3, 4, 5])
    image_points = image_points[selected_indices]
    object_points = object_points[selected_indices]

    # 初始化PnP求解器
    pnpsolver = PnPSolver.from_config(config_path)

    # 求解PnP
    success, R, tvec, residual = pnpsolver.solve(object_points, image_points)

    # 可视化结果
    if success:
        img = cv2.imread(image_path)
        vis_img = pnpsolver.draw_visualize_image(img)
        vis_img = cv2.resize(vis_img, (1536, 1024))
        cv2.imwrite("PnP.png", vis_img)
        print("[INFO] 投影图像已保存为 PnP.png")
        output_yaml = "pnp_result.yaml"
        fs = cv2.FileStorage(output_yaml, cv2.FILE_STORAGE_WRITE)

        fs.write("camera_matrix", pnpsolver.camera_matrix)
        fs.write("dist_coeffs", pnpsolver.dist_coeffs)
        fs.write("rotation_matrix", R)
        fs.write("translation_vector", tvec)
        fs.write("projection_residual", residual)

        fs.release()
        print(f"[INFO] 标定结果已保存到 {output_yaml}")
