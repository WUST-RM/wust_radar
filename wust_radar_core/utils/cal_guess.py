import cv2
import numpy as np
import yaml
import os

# 场地尺寸（厘米）
field_width_cm = 1500.0
field_height_cm = 2800.0

# 读取图像
image_path = "/home/hy/wust_radar/src/wust_radar_core/field_image.png"
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load image from {image_path}")
    exit(1)

# 保存无点的原始图像
image_backup = image.copy()

# 默认空的各机器人点位字典
guess_points = {
    "R1": [],
    "R2": [],
    "R3": [],
    "R4": [],
    "R7": [],
    "B1": [],
    "B2": [],
    "B3": [],
    "B4": [],
    "B7": [],
}

current_robot = "R1"
current_point_idx = 0  # 当前高亮点索引

guess_points_file = "/home/hy/wust_radar/src/wust_radar_core/config/guess_pts.yaml"
save_guess_points_file = "output/guess_points_output.yaml"

# 启动时尝试加载已有点位
if os.path.isfile(guess_points_file):
    with open(guess_points_file, "r") as f:
        data = yaml.safe_load(f)
        if data and "guess_points" in data:
            guess_points = data["guess_points"]
            print(f"Loaded guess points from {guess_points_file}")
else:
    print(f"No guess points file found, starting empty")


def image_to_world(px, py, image_shape):
    height, width = image_shape[:2]
    x_norm = 1 - (px / width)
    y_norm = 1 - (py / height)
    x_world = x_norm * field_height_cm
    y_world = y_norm * field_width_cm
    return x_world / 100.0, y_world / 100.0


def redraw_points():
    global image, current_point_idx
    image = image_backup.copy()
    points = guess_points.get(current_robot, [])

    for i, (wx, wy) in enumerate(points):
        px = int((1 - (wx * 100.0) / field_height_cm) * image.shape[1])
        py = int((1 - (wy * 100.0) / field_width_cm) * image.shape[0])
        if i == current_point_idx:
            color = (0, 0, 255)
            cv2.circle(image, (px, py), 8, color, -1)
        else:
            color = (0, 255, 0)
            cv2.circle(image, (px, py), 5, color, -1)
        cv2.putText(
            image,
            str(i),
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow("Image", image)


def mouse_callback(event, x, y, flags, param):
    global current_point_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        wx, wy = image_to_world(x, y, image.shape)
        print(f"Image point: ({x}, {y}) -> World point: ({wx:.2f} m, {wy:.2f} m)")
        guess_points.setdefault(current_robot, []).append([round(wx, 2), round(wy, 2)])
        current_point_idx = len(guess_points[current_robot]) - 1  # 新点变成当前点
        redraw_points()


class FlowList(list):
    pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)


def convert_to_flow_lists(d):
    if isinstance(d, list):
        return FlowList(convert_to_flow_lists(i) for i in d)
    elif isinstance(d, dict):
        return {k: convert_to_flow_lists(v) for k, v in d.items()}
    else:
        return d


def save_guess_points_to_yaml(filename=save_guess_points_file):
    converted = {k: convert_to_flow_lists(v) for k, v in guess_points.items()}
    with open(filename, "w") as f:
        yaml.dump(
            {"guess_points": converted}, f, default_flow_style=False, sort_keys=False
        )
    print(f"Guess points saved to {filename}")


print("当前选中机器人:", current_robot)
print(
    "点击左键添加点，按数字键1-9切换机器人，按左右箭头切换当前点，按u撤销当前点，按q退出并保存"
)

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_callback)
redraw_points()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key in [ord(str(i)) for i in range(0, 10)]:
        robot_map = {
            "1": "R1",
            "2": "R2",
            "3": "R3",
            "4": "R4",
            "5": "R7",
            "6": "B1",
            "7": "B2",
            "8": "B3",
            "9": "B4",
            "0": "B7",
        }
        current_robot = robot_map.get(chr(key), current_robot)
        current_point_idx = 0
        print("当前选中机器人:", current_robot)
        redraw_points()

    elif key == 81:  # 左箭头 VK_LEFT
        points = guess_points.get(current_robot, [])
        if points:
            current_point_idx = (current_point_idx - 1) % len(points)
            redraw_points()
            print(f"当前选中点索引: {current_point_idx}")
    elif key == 83:  # 右箭头 VK_RIGHT
        points = guess_points.get(current_robot, [])
        if points:
            current_point_idx = (current_point_idx + 1) % len(points)
            redraw_points()
            print(f"当前选中点索引: {current_point_idx}")

    elif key == ord("u"):  # 撤销当前选中点（而非最后一个）
        points = guess_points.get(current_robot, [])
        if points:
            removed = points.pop(current_point_idx)
            print(f"撤销点 {removed} from {current_robot} 索引 {current_point_idx}")
            # 撤销后调整 current_point_idx：
            # 如果当前点索引超出剩余点数量，退回到最后一个点
            if current_point_idx >= len(points):
                current_point_idx = max(0, len(points) - 1)
            redraw_points()
        else:
            print(f"{current_robot} 没有点可以撤销")

    elif key == ord("q"):  # 退出并保存
        save_guess_points_to_yaml()
        break

cv2.destroyAllWindows()
