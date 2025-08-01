import cv2
import yaml
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os

# —— 0. 参数文件路径 & 视频路径 ——————————————————————————————
yaml_path = "/home/hy/wust_radar/src/wust_binocular/config/camera_info.yaml"
input_path = "/home/hy/data/video_save/国科大1.avi"
left_path = "left.mp4"
right_path = "right.mp4"
stereo_path = "stereo.mp4"

# —— 1. 加载 YAML 中的相机参数 —————————————————————————————
with open(yaml_path, "r") as f:
    cam = yaml.safe_load(f)

K_data = cam["camera_matrix"]["data"]
K = np.array(K_data, dtype=np.float32).reshape(3, 3)
dist = np.array(cam["distortion_coefficients"]["data"], dtype=np.float32).reshape(1, -1)

# —— 2. 手动设置双目基线 ————————————————————————————————
baseline = 0.06  # 6 cm

# —— 3. 初始化 MiDaS 深度模型 ——————————————————————————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = (
    torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, revision="master")
    .to(device)
    .eval()
)

transform = Compose(
    [
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# —— 4. 视频读取 & 写入配置 ———————————————————————————————
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"无法打开视频文件: {input_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if frame_count == 0:
    print("⚠️ 警告：无法读取视频帧数，frame_count == 0，进度打印将不准确。")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer_left = cv2.VideoWriter(left_path, fourcc, fps, (w, h))
writer_right = cv2.VideoWriter(right_path, fourcc, fps, (w, h))
writer_stereo = cv2.VideoWriter(stereo_path, fourcc, fps, (w * 2, h))

map1, map2 = cv2.initUndistortRectifyMap(K, dist, np.eye(3), K, (w, h), cv2.CV_32FC1)


# —— 5. 合成右目图像 —————————————————————————————————————
def synthesize_right(frame, depth, K, baseline):
    j, i = np.indices(depth.shape, dtype=np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = np.maximum(depth, 1e-3)
    X = (i - cx) * Z / fx
    Y = (j - cy) * Z / fy
    Xr = X - baseline
    ur = (Xr * fx / Z + cx).astype(np.float32)
    vr = j
    return cv2.remap(
        frame,
        ur,
        vr,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# —— 6. 主循环：读取 → 去畸 → 深度估计 → 合成右目 —————————
with torch.no_grad():
    idx = 0
    while True:
        ret, raw = cap.read()
        if not ret:
            break
        idx += 1

        # 进度打印（每 30 帧一次）
        if idx % 30 == 0:
            print(f"[{idx}/{frame_count}] Processing...")

        # 左目图像
        left = cv2.remap(raw, map1, map2, interpolation=cv2.INTER_LINEAR)

        # 深度估计输入
        img_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 384, 384]

        pred = midas(input_tensor)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)  # [1, 1, H, W]

        # 上采样为原图大小
        depth = (
            torch.nn.functional.interpolate(
                pred, size=(h, w), mode="bicubic", align_corners=False
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        # 深度归一化（转换为 0.1m ~ 5.1m 范围）
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-6)
        depth = depth * 25.0 + 0.1

        # 合成右目图像
        right = synthesize_right(left, depth, K, baseline)

        # 写入视频
        stereo = np.hstack([left, right])
        writer_left.write(left)
        writer_right.write(right)
        writer_stereo.write(stereo)

        # 若无 GUI 显示环境，建议注释以下两行以避免 Qt 报错
        # cv2.imshow("Stereo Preview", cv2.resize(stereo, (w, h)))
        # if cv2.waitKey(1) == 27: break

# —— 7. 释放资源 ———————————————————————————————————————
cap.release()
writer_left.release()
writer_right.release()
writer_stereo.release()
cv2.destroyAllWindows()

print("✅ 处理完成！输出结果：")
print(f"  👁 左目视频：{left_path}")
print(f"  👁 右目视频：{right_path}")
print(f"  🎞 立体视频：{stereo_path}")
