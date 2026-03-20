from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/doyoung/Documents/Blender")
DOCS = ROOT / "docs"
DATASET = ROOT / "datasets" / "v3"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")

PANELS = [
    {"frame": 623, "image": DATASET / "images" / "image_0623.png", "label": DATASET / "labels" / "label_0623.json"},
    {"frame": 686, "image": DATASET / "images" / "image_0686.png", "label": DATASET / "labels" / "label_0686.json"},
    {"frame": 720, "image": DATASET / "images" / "image_0720.png", "label": DATASET / "labels" / "label_0720.json"},
]

OUT_PATH = DOCS / "3dof_storyboard_623_686_720.png"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def rounded_box(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=1, radius=24):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def line_with_arrow(draw: ImageDraw.ImageDraw, start, end, fill, width=5, arrow=14):
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    tip = (x1, y1)
    left = (x1 - arrow * ux + 0.6 * arrow * px, y1 - arrow * uy + 0.6 * arrow * py)
    right = (x1 - arrow * ux - 0.6 * arrow * px, y1 - arrow * uy - 0.6 * arrow * py)
    draw.polygon([tip, left, right], fill=fill)


def draw_multiline(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, line_gap: int = 6):
    x, y = xy
    for line in text.splitlines():
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap


def fit_image(src: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / src.width, max_h / src.height)
    size = (int(src.width * scale), int(src.height * scale))
    return src.resize(size, Image.Resampling.LANCZOS)


def project_point(point, scale, offset):
    return (offset[0] + point[0] * scale, offset[1] + point[1] * scale)


def make_storyboard():
    canvas_w, canvas_h = 2200, 1500
    bg = Image.new("RGB", (canvas_w, canvas_h), "#f4efe7")
    draw = ImageDraw.Draw(bg)

    font_title = load_font(44)
    font_sub = load_font(24)
    font_body = load_font(22)
    font_small = load_font(18)
    font_section = load_font(28)
    font_big = load_font(30)

    draw.text((80, 50), "3 자유도만 추론하면 나머지가 따라오는 이유", font=font_title, fill="#1e2728")
    draw.text(
        (80, 108),
        "실제 프레임 623, 686, 720을 사용한 발표용 시각화: 이미지에서 읽는 3개와 기하로 복원되는 값의 분리",
        font=font_sub,
        fill="#516063",
    )

    left_x = 70
    card_w = 1460
    card_h = 390
    img_box_w = 760
    img_box_h = 300
    card_gap = 30

    accent_red = "#c85c43"
    accent_teal = "#2f7f73"
    accent_gold = "#d19a2a"
    dark = "#223032"

    for idx, panel in enumerate(PANELS):
        top = 170 + idx * (card_h + card_gap)
        rounded_box(draw, (left_x, top, left_x + card_w, top + card_h), fill="#fffaf2", outline="#d9cdbd", width=2)

        with panel["label"].open("r", encoding="utf-8") as f:
            label = json.load(f)
        image = Image.open(panel["image"]).convert("RGB")
        fitted = fit_image(image, img_box_w, img_box_h)

        img_x = left_x + 28
        img_y = top + 58
        bg.paste(fitted, (img_x, img_y))
        draw.rounded_rectangle((img_x, img_y, img_x + fitted.width, img_y + fitted.height), radius=18, outline="#cabda8", width=2)

        scale = fitted.width / image.width
        foot = label["ground_truth"]["truck_center_2d"]
        origin = label["ground_truth"]["axes_2d"]["origin"]
        x_end = label["ground_truth"]["axes_2d"]["x_end"]
        corners = label["ground_truth"]["2d_corners"]

        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        bbox = (
            img_x + min(xs) * scale,
            img_y + min(ys) * scale,
            img_x + max(xs) * scale,
            img_y + max(ys) * scale,
        )
        draw.rounded_rectangle(bbox, radius=16, outline="#ffffff", width=4)

        foot_pt = project_point(foot, scale, (img_x, img_y))
        origin_pt = project_point(origin, scale, (img_x, img_y))
        x_end_pt = project_point(x_end, scale, (img_x, img_y))
        cy_y = img_y + 540 * scale

        draw.line([(img_x, cy_y), (img_x + fitted.width, cy_y)], fill=(255, 255, 255, 150), width=3)
        draw.ellipse((foot_pt[0] - 10, foot_pt[1] - 10, foot_pt[0] + 10, foot_pt[1] + 10), fill=accent_teal, outline="white", width=3)
        draw.line([(foot_pt[0], cy_y), foot_pt], fill=accent_red, width=5)
        line_with_arrow(draw, origin_pt, x_end_pt, fill=accent_gold, width=6, arrow=18)

        draw.text((img_x, top + 20), f"Frame {panel['frame']}", font=font_section, fill=dark)
        draw.text((img_x + 170, top + 24), f"view={label['view_category']}  h_cam={label['metadata']['h_cam']:.2f}m", font=font_small, fill="#647275")

        text_x = left_x + 830
        draw.text((text_x, top + 26), "이미지에서 직접 읽는 3개", font=font_big, fill=dark)

        bullet_y = top + 88
        bullets = [
            (accent_teal, "1. foot center (u, v)", "차량이 바닥과 만나는 점의 위치"),
            (accent_gold, "2. yaw", "차량이 어느 방향을 보는지"),
            ("#5970b2", "3. object center cue", "2D 위치가 좌우 위치와 연결됨"),
        ]
        for color, head, sub in bullets:
            draw.ellipse((text_x, bullet_y + 8, text_x + 14, bullet_y + 22), fill=color)
            draw.text((text_x + 26, bullet_y), head, font=font_body, fill=dark)
            draw.text((text_x + 26, bullet_y + 30), sub, font=font_small, fill="#5c6769")
            bullet_y += 72

        rounded_box(draw, (text_x, top + 260, text_x + 560, top + 344), fill="#f0f5f2", outline="#c8d7d0", width=2, radius=18)
        draw.text((text_x + 20, top + 278), "해석 포인트", font=font_body, fill=accent_teal)

        frame_note = {
            623: "foot point가 더 아래에 있어 상대적으로 멀리 있는 장면으로 읽힘",
            686: "yaw 단서가 강해서 박스의 회전 방향을 안정적으로 설명하기 좋음",
            720: "카메라가 낮고 대상이 가까워 v 변화가 depth 차이로 잘 보임",
        }[panel["frame"]]
        draw.text((text_x + 20, top + 308), frame_note, font=font_small, fill="#495658")

    right_x = 1590
    rounded_box(draw, (right_x, 170, 2130, 605), fill="#fffaf2", outline="#d9cdbd", width=2)
    draw.text((right_x + 26, 196), "이미 알고 있는 제약", font=font_big, fill=dark)
    known = [
        "카메라 높이 h_cam은 메타데이터로 이미 알려져 있음",
        "트럭의 발점은 항상 ground plane 위에 놓임",
        "트럭 크기(W/H/L)는 고정 prior 또는 아주 좁은 범위로 둠",
    ]
    ky = 255
    for item in known:
        draw.ellipse((right_x + 30, ky + 8, right_x + 42, ky + 20), fill=accent_teal)
        draw.text((right_x + 56, ky), item, font=font_small, fill="#475759")
        ky += 72

    rounded_box(draw, (right_x, 635, 2130, 1125), fill="#fffaf2", outline="#d9cdbd", width=2)
    draw.text((right_x + 26, 660), "그래서 따라오는 값", font=font_big, fill=dark)
    derived = [
        ("Depth Z", "foot point가 화면에서 얼마나 아래로 내려왔는지로 결정"),
        ("Lateral X", "u 위치와 Z가 정해지면 좌우 위치도 함께 정해짐"),
        ("3D Box", "yaw + 위치 + 고정 크기로 3D box 전체를 복원"),
    ]
    dy = 725
    for head, body in derived:
        rounded_box(draw, (right_x + 24, dy, 2106, dy + 106), fill="#f5f0e5", outline="#e0d3bc", width=1, radius=18)
        draw.text((right_x + 42, dy + 18), head, font=font_body, fill=accent_red)
        draw.text((right_x + 42, dy + 52), body, font=font_small, fill="#4e5b5d")
        dy += 124

    rounded_box(draw, (right_x, 1155, 2130, 1410), fill="#213133", outline="#213133", width=1)
    draw.text((right_x + 26, 1185), "발표용 한 문장", font=font_big, fill="#f8efe0")
    quote = (
        "이 장면에서는 6자유도를 모두 네트워크가 새로 발명할 필요가 없습니다.\n"
        "이미지에서 필요한 것은 발 위치(u, v)와 방향(yaw)뿐이고,\n"
        "카메라 높이와 ground contact 제약이 깊이와 나머지 위치를 자동으로 묶어줍니다."
    )
    draw_multiline(draw, (right_x + 28, 1245), quote, font_small, "#f8efe0", line_gap=10)

    bg.save(OUT_PATH, quality=95)


if __name__ == "__main__":
    make_storyboard()
