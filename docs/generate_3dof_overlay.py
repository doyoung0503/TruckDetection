from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/doyoung/Documents/Blender")
DATASET = ROOT / "datasets" / "v3"
DOCS = ROOT / "docs"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")

FRAMES = [623, 686, 720]

C_BOX_REAR = "#72a7ff"
C_BOX_FRONT = "#50cf73"
C_BOX_PILLAR = "#e7e7e7"
C_FOOT = "#27c2b5"
C_CENTER = "#ffd166"
C_YAW = "#ffb703"
C_DEPTH = "#ef476f"
C_V = "#0ea5e9"
C_TEXT = "#f8f8f6"
C_MUTED = "#d4d4cf"
C_PANEL = (18, 23, 28, 215)
C_PANEL_LIGHT = "#f4efe5"
C_OUTLINE = "#d9cdbd"

REAR_FACE = [0, 1, 2, 3]
FRONT_FACE = [4, 5, 6, 7]
PILLARS = [(0, 4), (1, 5), (2, 6), (3, 7)]


def load_font(size: int):
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def pt(xy):
    return (int(round(xy[0])), int(round(xy[1])))


def draw_dashed_line(draw: ImageDraw.ImageDraw, p1, p2, fill, width=3, dash=14, gap=9):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    draw_on = True
    while pos < length:
        step = dash if draw_on else gap
        end = min(length, pos + step)
        if draw_on:
            a = (x1 + ux * pos, y1 + uy * pos)
            b = (x1 + ux * end, y1 + uy * end)
            draw.line([pt(a), pt(b)], fill=fill, width=width)
        pos += step
        draw_on = not draw_on


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, fill, width=4, head=18):
    draw.line([pt(start), pt(end)], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    norm = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    left = (x1 - head * ux + 0.55 * head * px, y1 - head * uy + 0.55 * head * py)
    right = (x1 - head * ux - 0.55 * head * px, y1 - head * uy - 0.55 * head * py)
    draw.polygon([pt((x1, y1)), pt(left), pt(right)], fill=fill)


def draw_cross(draw: ImageDraw.ImageDraw, center, size, fill, width=4):
    cx, cy = center
    draw.line([pt((cx - size, cy)), pt((cx + size, cy))], fill=fill, width=width)
    draw.line([pt((cx, cy - size)), pt((cx, cy + size))], fill=fill, width=width)
    draw.ellipse((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), fill=fill)


def face_edge(draw: ImageDraw.ImageDraw, corners, indices, fill, width=4):
    for i, j in zip(indices, indices[1:] + [indices[0]]):
        draw.line([pt(corners[i][:2]), pt(corners[j][:2])], fill=fill, width=width)


def multiline(draw: ImageDraw.ImageDraw, xy, lines, font, fill, gap=6):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + gap


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=1, radius=22):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def label_path(frame: int) -> Path:
    return DATASET / "labels" / f"label_{frame:04d}.json"


def image_path(frame: int) -> Path:
    return DATASET / "images" / f"image_{frame:04d}.png"


def render_frame(frame: int):
    with label_path(frame).open("r", encoding="utf-8") as f:
        lbl = json.load(f)

    img = Image.open(image_path(frame)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    font_title = load_font(32)
    font_body = load_font(22)
    font_small = load_font(18)
    font_tiny = load_font(16)

    gt = lbl["ground_truth"]
    md = lbl["metadata"]
    dims = lbl["truck_dims"]
    corners = gt["2d_corners"]
    foot_idx = [0, 1, 4, 5]

    foot = (
        sum(corners[i][0] for i in foot_idx) / 4.0,
        sum(corners[i][1] for i in foot_idx) / 4.0,
    )
    center_2d = tuple(gt["truck_center_2d"])
    x_end = tuple(gt["axes_2d"]["x_end"])

    fx = md["K_matrix"][0][0]
    fy = md["K_matrix"][1][1]
    cx = md["K_matrix"][0][2]
    cy = md["K_matrix"][1][2]
    h_cam = md["h_cam"]
    dv = foot[1] - cy
    z = fy * h_cam / max(dv, 1e-6)
    x = (foot[0] - cx) * z / fx
    y_foot = h_cam
    y_center = h_cam - dims["height"] / 2.0

    draw.rectangle((0, 0, w, 98), fill=(18, 23, 28, 185))
    draw.text((34, 24), f"Frame {frame}: 실제 이미지 위에서 3DoF가 나머지 자유도를 결정하는 과정", font=font_title, fill=C_TEXT)
    draw.text((34, 63), "3D box, foot center, yaw, depth, 3D center Y를 한 장에 겹쳐서 표시", font=font_small, fill=C_MUTED)

    face_edge(draw, corners, REAR_FACE, C_BOX_REAR, width=5)
    face_edge(draw, corners, FRONT_FACE, C_BOX_FRONT, width=5)
    for i, j in PILLARS:
        draw.line([pt(corners[i][:2]), pt(corners[j][:2])], fill=C_BOX_PILLAR, width=4)

    ymin = min(p[1] for p in corners)
    ymax = max(p[1] for p in corners)
    xmin = min(p[0] for p in corners)
    xmax = max(p[0] for p in corners)

    draw_cross(draw, foot, 14, C_FOOT)
    draw_cross(draw, center_2d, 12, C_CENTER)

    draw_dashed_line(draw, (foot[0], cy), foot, C_V, width=4)
    draw.line([pt((cx, 0)), pt((cx, h))], fill=(130, 150, 155, 110), width=2)
    draw.line([pt((0, cy)), pt((w, cy))], fill=(130, 150, 155, 110), width=2)

    yaw_len = 140
    yaw_dir = (foot[0] + (x_end[0] - center_2d[0]) * 0.9, foot[1] + (x_end[1] - center_2d[1]) * 0.9)
    dx = yaw_dir[0] - foot[0]
    dy = yaw_dir[1] - foot[1]
    n = max(math.hypot(dx, dy), 1e-6)
    yaw_end = (foot[0] + yaw_len * dx / n, foot[1] + yaw_len * dy / n)
    draw_arrow(draw, foot, yaw_end, C_YAW, width=6, head=22)

    label_specs = [
        ((foot[0] + 26, foot[1] - 64), "foot center (u, v)\n이 점만 ground에 닿음", C_FOOT),
        ((center_2d[0] + 28, center_2d[1] - 34), "3D box center의 2D 투영", C_CENTER),
        ((yaw_end[0] + 18, yaw_end[1] - 30), "yaw\n차량 방향", C_YAW),
        ((foot[0] + 24, (foot[1] + cy) / 2 - 18), "v - cy\n화면에서 내려온 양", C_V),
    ]

    for (tx, ty), text, color in label_specs:
        bbox = draw.multiline_textbbox((tx, ty), text, font=font_small, spacing=4)
        pad = 10
        rounded(draw, (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(18, 23, 28, 208), outline=color, width=2, radius=16)
        draw.multiline_text((tx, ty), text, font=font_small, fill=C_TEXT, spacing=4)

    # 3D box height cue
    box_mid_x = xmax + 36
    draw.line([pt((box_mid_x, ymin)), pt((box_mid_x, ymax))], fill=(255, 255, 255, 180), width=3)
    draw.line([pt((box_mid_x - 11, ymin)), pt((box_mid_x + 11, ymin))], fill=(255, 255, 255, 220), width=3)
    draw.line([pt((box_mid_x - 11, ymax)), pt((box_mid_x + 11, ymax))], fill=(255, 255, 255, 220), width=3)
    ht_text = f"known H = {dims['height']:.2f} m"
    bbox = draw.textbbox((box_mid_x + 18, (ymin + ymax) / 2 - 10), ht_text, font=font_small)
    rounded(draw, (bbox[0] - 8, bbox[1] - 8, bbox[2] + 8, bbox[3] + 8), fill=(18, 23, 28, 208), outline="#ffffff", width=2, radius=14)
    draw.text((box_mid_x + 18, (ymin + ymax) / 2 - 10), ht_text, font=font_small, fill=C_TEXT)

    # Bottom explanation panel
    panel = (34, h - 276, 1188, h - 38)
    rounded(draw, panel, fill=C_PANEL, outline=(255, 255, 255, 55), width=2, radius=26)
    draw.text((58, h - 246), "이 이미지에서 직접 읽는 값과 기하로 확정되는 값", font=font_body, fill=C_TEXT)
    lines = [
        f"직접 읽는 값: foot center=({foot[0]:.1f}, {foot[1]:.1f}), yaw={gt['yaw_theta']:.1f}deg",
        f"이미 알고 있는 값: h_cam={h_cam:.2f}m, truck height={dims['height']:.2f}m, width={dims['width']:.2f}m, length={dims['length']:.2f}m",
        f"depth Z는 foot point가 principal line 아래로 내려온 양 dv={dv:.1f}px 때문에 결정됨",
        f"3D center Y는 bottom center Y_foot={y_foot:.2f}m 에서 높이의 절반을 올려 Y_center={y_center:.2f}m 로 정해짐",
    ]
    multiline(draw, (58, h - 206), lines, font_small, C_MUTED, gap=8)

    # Side inset explaining Z and Y_center.
    inset = (1268, h - 370, w - 34, h - 38)
    rounded(draw, inset, fill=(244, 239, 229, 240), outline=C_OUTLINE, width=2, radius=26)
    draw.text((1294, h - 342), "측면도 인셋: 왜 Z와 Y_center가 남는 자유도가 아닌가", font=font_body, fill="#1f2a2c")

    ix0, iy0, ix1, iy1 = inset
    base_y = iy1 - 70
    cam_x = ix0 + 70
    truck_x = ix0 + 320
    scale_y = 95

    draw.line([pt((ix0 + 24, base_y)), pt((ix1 - 24, base_y))], fill="#6a7b7d", width=3)
    draw.text((ix0 + 28, base_y + 12), "ground plane", font=font_tiny, fill="#4b5a5c")

    cam_y = base_y - h_cam * scale_y
    draw.line([pt((cam_x, base_y)), pt((cam_x, cam_y))], fill=C_V, width=4)
    draw.ellipse((cam_x - 12, cam_y - 12, cam_x + 12, cam_y + 12), fill="#1f2937")
    draw.text((cam_x - 26, cam_y - 44), "camera", font=font_tiny, fill="#1f2937")
    draw.text((cam_x + 12, (cam_y + base_y) / 2 - 10), f"h_cam={h_cam:.2f}m", font=font_tiny, fill=C_V)

    foot_world = (truck_x, base_y)
    top_world = (truck_x, base_y - dims["height"] * scale_y)
    center_world = (truck_x, base_y - dims["height"] * scale_y / 2)
    draw.line([pt(foot_world), pt(top_world)], fill=C_BOX_FRONT, width=10)
    draw.ellipse((foot_world[0] - 10, foot_world[1] - 10, foot_world[0] + 10, foot_world[1] + 10), fill=C_FOOT)
    draw.ellipse((center_world[0] - 10, center_world[1] - 10, center_world[0] + 10, center_world[1] + 10), fill=C_CENTER)
    draw.text((truck_x + 18, foot_world[1] - 20), f"bottom center Y={y_foot:.2f}m", font=font_tiny, fill=C_FOOT)
    draw.text((truck_x + 18, center_world[1] - 20), f"3D center Y={y_center:.2f}m", font=font_tiny, fill=C_CENTER)
    draw.text((truck_x + 18, top_world[1] - 20), f"H={dims['height']:.2f}m", font=font_tiny, fill=C_BOX_FRONT)

    draw_arrow(draw, (cam_x + 18, cam_y + 6), (truck_x - 20, foot_world[1] - 10), C_DEPTH, width=5, head=20)
    draw.text((ix0 + 165, iy0 + 120), f"depth Z={z:.2f}m", font=font_body, fill=C_DEPTH)
    inset_lines = [
        "1. foot center는 ground plane 위에 있으므로 bottom center의 Y는 h_cam으로 고정",
        "2. 이미지에서 v가 정해지면 principal point와의 차이로 depth Z가 정해짐",
        "3. 객체 높이를 알고 있으므로 3D center Y는 bottom center에서 H/2만큼 위",
    ]
    multiline(draw, (1294, iy0 + 160), inset_lines, font_tiny, "#495658", gap=7)

    out = DOCS / f"3dof_overlay_{frame:04d}.png"
    img.save(out, quality=95)
    return out


def make_contact_sheet(paths: list[Path]):
    images = [Image.open(p).convert("RGB") for p in paths]
    thumb_w = 1100
    thumb_h = int(images[0].height * thumb_w / images[0].width)
    canvas = Image.new("RGB", (thumb_w, thumb_h * len(images)), "#f4efe5")
    for i, img in enumerate(images):
        thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (0, i * thumb_h))
    out = DOCS / "3dof_overlay_contact_sheet.png"
    canvas.save(out, quality=92)
    return out


def main():
    outputs = [render_frame(frame) for frame in FRAMES]
    make_contact_sheet(outputs)


if __name__ == "__main__":
    main()
