from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("/Users/doyoung/Documents/Blender")
DATASET = ROOT / "datasets" / "v3"
DOCS = ROOT / "docs"

FRAMES = [623, 686, 720]

REAR_FACE = [0, 1, 2, 3]
FRONT_FACE = [4, 5, 6, 7]
PILLARS = [(0, 4), (1, 5), (2, 6), (3, 7)]

C_REAR = "#71a8ff"
C_FRONT = "#4fd17a"
C_PILLAR = "#ececec"
C_FOOT = "#2dd4bf"
C_CENTER = "#ffd166"
C_YAW = "#ffb703"
C_DV = "#38bdf8"
C_DEPTH = "#ef476f"
C_HEIGHT = "#ffffff"
C_PANEL = (247, 241, 231, 238)
C_PANEL_STROKE = "#d8cbbb"
C_GROUND = "#6b7b7d"
C_CAMERA = "#1f2937"


def pt(xy):
    return (int(round(xy[0])), int(round(xy[1])))


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=1, radius=24):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, fill, width=5, head=18):
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


def draw_dashed_line(draw: ImageDraw.ImageDraw, p1, p2, fill, width=4, dash=16, gap=10):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    on = True
    while pos < length:
        step = dash if on else gap
        end = min(length, pos + step)
        if on:
            a = (x1 + ux * pos, y1 + uy * pos)
            b = (x1 + ux * end, y1 + uy * end)
            draw.line([pt(a), pt(b)], fill=fill, width=width)
        pos += step
        on = not on


def draw_cross(draw: ImageDraw.ImageDraw, center, size, fill, width=4):
    cx, cy = center
    draw.line([pt((cx - size, cy)), pt((cx + size, cy))], fill=fill, width=width)
    draw.line([pt((cx, cy - size)), pt((cx, cy + size))], fill=fill, width=width)
    draw.ellipse((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), fill=fill)


def draw_ring(draw: ImageDraw.ImageDraw, center, radius, fill, width=5):
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=fill, width=width)


def draw_bracket(draw: ImageDraw.ImageDraw, x, y0, y1, fill, width=4, tick=16):
    draw.line([pt((x, y0)), pt((x, y1))], fill=fill, width=width)
    draw.line([pt((x - tick, y0)), pt((x + tick, y0))], fill=fill, width=width)
    draw.line([pt((x - tick, y1)), pt((x + tick, y1))], fill=fill, width=width)


def face(draw: ImageDraw.ImageDraw, corners, indices, fill, width=5):
    for i, j in zip(indices, indices[1:] + [indices[0]]):
        draw.line([pt(corners[i][:2]), pt(corners[j][:2])], fill=fill, width=width)


def dot_chain(draw: ImageDraw.ImageDraw, start, end, fill, count=5, r=6):
    x0, y0 = start
    x1, y1 = end
    for i in range(count):
        t = i / (count - 1) if count > 1 else 0
        x = x0 * (1 - t) + x1 * t
        y = y0 * (1 - t) + y1 * t
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def triangle_marker(draw: ImageDraw.ImageDraw, center, size, fill):
    cx, cy = center
    pts = [(cx, cy - size), (cx - 0.9 * size, cy + size), (cx + 0.9 * size, cy + size)]
    draw.polygon([pt(p) for p in pts], fill=fill)


def label_path(frame: int) -> Path:
    return DATASET / "labels" / f"label_{frame:04d}.json"


def image_path(frame: int) -> Path:
    return DATASET / "images" / f"image_{frame:04d}.png"


def render_frame(frame: int) -> Path:
    with label_path(frame).open("r", encoding="utf-8") as f:
        lbl = json.load(f)

    img = Image.open(image_path(frame)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    gt = lbl["ground_truth"]
    md = lbl["metadata"]
    dims = lbl["truck_dims"]
    corners = gt["2d_corners"]
    axes = gt["axes_2d"]
    foot_idx = [0, 1, 4, 5]

    foot = (
        sum(corners[i][0] for i in foot_idx) / 4.0,
        sum(corners[i][1] for i in foot_idx) / 4.0,
    )
    center_2d = tuple(gt["truck_center_2d"])
    x_end = tuple(axes["x_end"])
    fx = md["K_matrix"][0][0]
    fy = md["K_matrix"][1][1]
    cx = md["K_matrix"][0][2]
    cy = md["K_matrix"][1][2]
    h_cam = md["h_cam"]
    dv = foot[1] - cy
    z = fy * h_cam / max(dv, 1e-6)

    face(draw, corners, REAR_FACE, C_REAR)
    face(draw, corners, FRONT_FACE, C_FRONT)
    for i, j in PILLARS:
        draw.line([pt(corners[i][:2]), pt(corners[j][:2])], fill=C_PILLAR, width=4)

    draw.line([pt((cx, 0)), pt((cx, h))], fill=(170, 177, 182, 90), width=2)
    draw.line([pt((0, cy)), pt((w, cy))], fill=(170, 177, 182, 90), width=2)

    draw_cross(draw, foot, 15, C_FOOT)
    draw_ring(draw, center_2d, 13, C_CENTER, width=4)
    triangle_marker(draw, (cx, cy - 24), 12, C_DV)

    draw_dashed_line(draw, (foot[0], cy), foot, C_DV, width=4)
    dot_chain(draw, (foot[0], cy + 24), (foot[0], foot[1] - 24), C_DV, count=6, r=5)

    dx = x_end[0] - center_2d[0]
    dy = x_end[1] - center_2d[1]
    norm = max(math.hypot(dx, dy), 1e-6)
    yaw_end = (foot[0] + 150 * dx / norm, foot[1] + 150 * dy / norm)
    draw_arrow(draw, foot, yaw_end, C_YAW, width=6, head=22)

    ymin = min(p[1] for p in corners)
    ymax = max(p[1] for p in corners)
    xmax = max(p[0] for p in corners)
    draw_bracket(draw, xmax + 34, ymin, ymax, C_HEIGHT, width=4, tick=16)

    # small cue circles instead of text labels
    cue_points = [
        (foot[0] + 48, foot[1] - 28, C_FOOT),
        (center_2d[0] + 38, center_2d[1] - 34, C_CENTER),
        (yaw_end[0] + 24, yaw_end[1] - 8, C_YAW),
        (foot[0] + 28, (foot[1] + cy) / 2, C_DV),
        (xmax + 65, (ymin + ymax) / 2, C_HEIGHT),
    ]
    for x, y, color in cue_points:
        draw_ring(draw, (x, y), 11, color, width=4)

    # bottom-left symbolic panel
    panel = (34, h - 280, 1140, h - 34)
    rounded(draw, panel, fill=(18, 23, 28, 210), outline=(255, 255, 255, 40), width=2, radius=28)

    px0, py0 = panel[0] + 42, panel[1] + 48
    # observed: foot + yaw
    draw_cross(draw, (px0 + 40, py0 + 54), 14, C_FOOT)
    draw_arrow(draw, (px0 + 110, py0 + 54), (px0 + 220, py0 + 54), C_YAW, width=6, head=20)
    draw.ellipse((px0 + 260, py0 + 34, px0 + 292, py0 + 66), fill=(255, 255, 255, 230))

    # transition arrow
    draw_arrow(draw, (px0 + 340, py0 + 54), (px0 + 470, py0 + 54), "#ffffff", width=5, head=20)

    # derived: depth + centerY + full box
    draw_arrow(draw, (px0 + 540, py0 + 12), (px0 + 625, py0 + 96), C_DEPTH, width=6, head=22)
    draw.line([pt((px0 + 700, py0 + 8)), pt((px0 + 700, py0 + 100))], fill=C_FOOT, width=8)
    draw.ellipse((px0 + 690, py0 + 90, px0 + 710, py0 + 110), fill=C_FOOT)
    draw.ellipse((px0 + 690, py0 + 44, px0 + 710, py0 + 64), fill=C_CENTER)
    box_x = px0 + 790
    box_y = py0 + 8
    draw.rectangle((box_x, box_y + 18, box_x + 120, box_y + 100), outline=C_FRONT, width=5)
    draw.rectangle((box_x + 34, box_y, box_x + 154, box_y + 82), outline=C_REAR, width=5)
    for a, b in [((box_x, box_y + 18), (box_x + 34, box_y)),
                 ((box_x + 120, box_y + 18), (box_x + 154, box_y)),
                 ((box_x, box_y + 100), (box_x + 34, box_y + 82)),
                 ((box_x + 120, box_y + 100), (box_x + 154, box_y + 82))]:
        draw.line([pt(a), pt(b)], fill=C_PILLAR, width=4)

    # right inset without text
    inset = (1260, h - 390, w - 34, h - 34)
    rounded(draw, inset, fill=C_PANEL, outline=C_PANEL_STROKE, width=2, radius=28)
    ix0, iy0, ix1, iy1 = inset
    base_y = iy1 - 62
    cam_x = ix0 + 76
    truck_x = ix0 + 324
    scale_y = 92
    obj_h = dims["height"] * scale_y

    draw.line([pt((ix0 + 26, base_y)), pt((ix1 - 26, base_y))], fill=C_GROUND, width=4)
    cam_y = base_y - h_cam * scale_y
    draw.line([pt((cam_x, base_y)), pt((cam_x, cam_y))], fill=C_DV, width=5)
    draw.ellipse((cam_x - 12, cam_y - 12, cam_x + 12, cam_y + 12), fill=C_CAMERA)
    triangle_marker(draw, (cam_x, base_y + 4), 11, C_DV)

    foot_world = (truck_x, base_y)
    center_world = (truck_x, base_y - obj_h / 2)
    top_world = (truck_x, base_y - obj_h)
    draw.line([pt(foot_world), pt(top_world)], fill=C_FRONT, width=10)
    draw.ellipse((foot_world[0] - 11, foot_world[1] - 11, foot_world[0] + 11, foot_world[1] + 11), fill=C_FOOT)
    draw.ellipse((center_world[0] - 11, center_world[1] - 11, center_world[0] + 11, center_world[1] + 11), fill=C_CENTER)
    draw_bracket(draw, truck_x + 42, top_world[1], foot_world[1], C_HEIGHT, width=4, tick=14)

    draw_arrow(draw, (cam_x + 16, cam_y + 6), (truck_x - 14, foot_world[1] - 10), C_DEPTH, width=6, head=20)
    draw.line([pt((center_world[0] - 66, center_world[1])), pt((center_world[0] + 66, center_world[1]))], fill=(255, 209, 102, 180), width=3)

    # tiny symbolic legend strip with only shapes
    legend_y = iy0 + 34
    symbols = [
        ("cross", C_FOOT),
        ("ring", C_CENTER),
        ("arrow", C_YAW),
        ("dots", C_DV),
        ("diag", C_DEPTH),
        ("brace", C_HEIGHT),
    ]
    sx = ix0 + 42
    for kind, color in symbols:
        if kind == "cross":
            draw_cross(draw, (sx, legend_y), 10, color, width=4)
        elif kind == "ring":
            draw_ring(draw, (sx, legend_y), 10, color, width=4)
        elif kind == "arrow":
            draw_arrow(draw, (sx - 18, legend_y), (sx + 18, legend_y), color, width=5, head=14)
        elif kind == "dots":
            dot_chain(draw, (sx - 16, legend_y), (sx + 16, legend_y), color, count=4, r=4)
        elif kind == "diag":
            draw_arrow(draw, (sx - 14, legend_y - 14), (sx + 14, legend_y + 14), color, width=5, head=14)
        elif kind == "brace":
            draw_bracket(draw, sx, legend_y - 14, legend_y + 14, color, width=4, tick=8)
        sx += 62

    out = DOCS / f"3dof_visual_notext_{frame:04d}.png"
    img.save(out, quality=95)
    return out


def make_contact_sheet(paths: list[Path]) -> Path:
    images = [Image.open(p).convert("RGB") for p in paths]
    thumb_w = 1100
    thumb_h = int(images[0].height * thumb_w / images[0].width)
    canvas = Image.new("RGB", (thumb_w, thumb_h * len(images)), "#f4efe5")
    for i, img in enumerate(images):
        thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (0, i * thumb_h))
    out = DOCS / "3dof_visual_notext_contact_sheet.png"
    canvas.save(out, quality=92)
    return out


def main():
    outputs = [render_frame(frame) for frame in FRAMES]
    make_contact_sheet(outputs)


if __name__ == "__main__":
    main()
