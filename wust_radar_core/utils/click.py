import datetime
import cv2
import json
import os
import numpy as np

clicked_points = []
zoom_factor = 2.0
zoom_center = None
subpix_win_size = (5, 5)
subpix_zero_zone = (-1, -1)


def on_mouse_click(event, x, y, flags, param):
    global clicked_points, img_gray, zoom_center
    if event == cv2.EVENT_LBUTTONDOWN:
        pt = np.array([[x, y]], dtype=np.float32)
        cv2.cornerSubPix(
            img_gray,
            pt,
            subpix_win_size,
            subpix_zero_zone,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        refined_pt = (float(pt[0][0]), float(pt[0][1]))
        clicked_points.append(refined_pt)
        zoom_center = refined_pt  # 点击点后更新放大中心
        print(f"[CLICK] 亚像素坐标: {refined_pt}")


def save_points_to_json(points, output_path):
    data = {str(i + 1): [float(pt[0]), float(pt[1])] for i, pt in enumerate(points)}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] 保存标注点到 {output_path}")


def zoom_on_point(img, center, zoom_factor=2.0, size=200):
    h, w = img.shape[:2]
    x, y = center
    x = int(round(x))
    y = int(round(y))

    half_size = int(size // (2 * zoom_factor))
    x1 = max(x - half_size, 0)
    y1 = max(y - half_size, 0)
    x2 = min(x + half_size, w - 1)
    y2 = min(y + half_size, h - 1)

    cropped = img[y1:y2, x1:x2]

    if cropped.size == 0:
        return None

    zoomed = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.line(
        zoomed, (size // 2 - 10, size // 2), (size // 2 + 10, size // 2), (0, 0, 255), 1
    )
    cv2.line(
        zoomed, (size // 2, size // 2 - 10), (size // 2, size // 2 + 10), (0, 0, 255), 1
    )
    return zoomed


def main():
    global img_gray, clicked_points, zoom_center, zoom_factor

    video_path = "/home/hy/data/ENTERPRIZE战队RM2025开源第一视角/7.27-华东理工.mp4"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 无法打开视频")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 视频总帧数: {total_frames}")

    def nothing(x):
        pass

    cv2.namedWindow("Select Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Frame", 640, 480)
    cv2.createTrackbar("Frame", "Select Frame", 0, total_frames - 1, nothing)

    image_path = None
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    while True:
        frame_id = cv2.getTrackbarPos("Frame", "Select Frame")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 无法读取该帧")
            break

        cv2.imshow("Select Frame", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):

            image_path = os.path.join(output_dir, f"{now_str}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"[INFO] 已保存帧图像: {image_path}")
            break
        elif key == 27:
            print("[INFO] 用户取消")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if image_path is None:
        print("[ERROR] 没有保存任何帧，退出")
        return

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clicked_points.clear()
    zoom_center = None
    zoom_factor = 2.0

    print("[INFO] 请点击图像添加点，按 u 撤回上一个点，按 q 结束标注")
    print("使用 + / - 键放大或缩小局部查看，方向键微调最后一个点位置")

    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate", 800, 600)
    cv2.setMouseCallback("Annotate", on_mouse_click)

    while True:
        display_img = img.copy()
        for pt in clicked_points:
            cv2.circle(
                display_img, (int(round(pt[0])), int(round(pt[1]))), 5, (0, 255, 0), -1
            )

        if clicked_points and zoom_center is not None:
            zoom_img = zoom_on_point(img, zoom_center, zoom_factor=zoom_factor)
            if zoom_img is not None:
                cv2.imshow("Zoom", zoom_img)

        cv2.imshow("Annotate", display_img)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("u"):
            if clicked_points:
                removed = clicked_points.pop()
                print(f"[UNDO] 撤回点: {removed}")
                zoom_center = clicked_points[-1] if clicked_points else None
        elif key == ord("+") or key == ord("="):
            zoom_factor = min(zoom_factor + 0.5, 10.0)
            print(f"[ZOOM] 放大倍数: {zoom_factor}")
        elif key == ord("-") or key == ord("_"):
            zoom_factor = max(zoom_factor - 0.5, 1.0)
            print(f"[ZOOM] 缩小倍数: {zoom_factor}")
        elif key in [81, 82, 83, 84]:  # 方向键左上右下
            if clicked_points:
                x, y = clicked_points[-1]
                step = 0.5
                if key == 81:
                    x -= step
                elif key == 82:
                    y -= step
                elif key == 83:
                    x += step
                elif key == 84:
                    y += step
                x = min(max(x, 0), img.shape[1] - 1)
                y = min(max(y, 0), img.shape[0] - 1)
                clicked_points[-1] = (x, y)
                zoom_center = (x, y)
                print(f"[ADJUST] 最后一个点微调到 {(x, y)}")
        elif key == 27:
            print("[INFO] 用户取消")
            return

    cv2.destroyAllWindows()

    json_path = os.path.join(output_dir, f"{now_str}.json")
    save_points_to_json(clicked_points, json_path)
    print(f"[INFO] 已保存标注点到: {json_path}")


if __name__ == "__main__":
    main()
