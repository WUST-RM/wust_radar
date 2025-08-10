import json
import numpy as np
import cv2
from typing import Tuple, Optional


class PnPSolver:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        verbose=False,
        R_init: Optional[np.ndarray] = None,
        t_init: Optional[np.ndarray] = None,
        residual_init: Optional[float] = None,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.verbose = verbose
        self.rvec = None
        self.tvec = None
        self.projected_points = None
        self.image_point = None
        self.R_init = R_init
        self.t_init = t_init
        self.residual_init = residual_init

    @classmethod
    def from_opencv_yaml(cls, yaml_path: str) -> "PnPSolver":
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"无法打开 YAML 文件: {yaml_path}")

        try:
            camera_matrix = fs.getNode("camera_matrix").mat()
            dist_coeffs = fs.getNode("dist_coeffs").mat()
            R = fs.getNode("rotation_matrix").mat()
            t = fs.getNode("translation_vector").mat()
            residual = fs.getNode("projection_residual").real()
        except Exception as e:
            fs.release()
            raise ValueError(f"读取 YAML 文件 {yaml_path} 时出错: {e}")

        fs.release()

        return cls(
            camera_matrix=camera_matrix.astype(np.float32),
            dist_coeffs=dist_coeffs.astype(np.float32),
            verbose=True,
            R_init=R,
            t_init=t,
            residual_init=residual,
        )

    def solve(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            raise ValueError("PnP 至少需要 4 对匹配点")
        if object_points.shape[0] != image_points.shape[0]:
            raise ValueError("3D 点与 2D 点数量不匹配")

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

    config_path = "/home/hy/wust_radar/src/wust_radar_core/config/cal_result.yaml"
    image_path = "/home/hy/wust_radar/src/wust_radar_core/output/20250810_153131.jpg"
    json_path = "/home/hy/wust_radar/src/wust_radar_core/output/20250810_153131.json"
    keypoints_3d_path = "/home/hy/wust_radar/src/wust_radar_core/field/keypoint_6.txt"

    # 加载2D点
    with open(json_path, "r") as f:
        image_points_dict = json.load(f)
    image_points = np.array(
        [image_points_dict[str(i)] for i in range(1, len(image_points_dict) + 1)],
        dtype=np.float32,
    )

    # 加载3D点
    object_points = np.loadtxt(keypoints_3d_path, dtype=np.float32)

    # 只取前 6 个点
    selected_indices = np.array([0, 1, 2, 3, 4, 5])
    image_points = image_points[selected_indices]
    object_points = object_points[selected_indices]

    # 初始化PnP求解器
    pnpsolver = PnPSolver.from_opencv_yaml(config_path)

    # 求解PnP
    success, R, tvec, residual = pnpsolver.solve(object_points, image_points)

    # 可视化结果
    if success:
        img = cv2.imread(image_path)
        vis_img = pnpsolver.draw_visualize_image(img)
        vis_img = cv2.resize(vis_img, (1536, 1024))
        cv2.imwrite("output/PnP.png", vis_img)
        print("[INFO] 投影图像已保存为 PnP.png")

        # 保存成 OpenCV YAML
        output_yaml = "output/pnp_result.yaml"
        fs = cv2.FileStorage(output_yaml, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", pnpsolver.camera_matrix)
        fs.write("dist_coeffs", pnpsolver.dist_coeffs)
        fs.write("rotation_matrix", R)
        fs.write("translation_vector", tvec)
        fs.write("projection_residual", residual)
        fs.release()

        print(f"[INFO] 标定结果已保存到 {output_yaml}")
