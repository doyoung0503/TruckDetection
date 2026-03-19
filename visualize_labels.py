#!/usr/bin/env python3
"""
visualize_labels.py
====================
생성된 합성 데이터셋의 레이블을 이미지에 시각화합니다.

출력 항목:
  - 3D 바운딩 박스 (visibility 반영: 실선/점선/숨김)
  - 트럭 기준점 십자 마커 (기하 중앙)
  - XYZ 축 화살표 (X=빨강, Y=초록/전진, Z=파랑/상향)
  - 우측 정보 패널 (거리, cam_pos, yaw, visibility 통계 등)

실행:
    python3 visualize_labels.py                        # v1 전체 시각화
    python3 visualize_labels.py --version v0_beta --num 5
    python3 visualize_labels.py --idx 3                # 특정 인덱스
    python3 visualize_labels.py --test rear            # test_rear.json/png
    python3 visualize_labels.py --test all             # 4방향 테스트 전부
"""

import json, os, math, argparse
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 색상 ──────────────────────────────────────────────────────────────────────
C_REAR   = (100, 149, 237)
C_FRONT  = ( 50, 205,  50)
C_PILLAR = (200, 200, 200)
C_OCCL   = (120, 120, 120)
C_CENTER = (255, 215,   0)
C_AXIS_X = (255,  60,  60)
C_AXIS_Y = ( 60, 220,  60)
C_AXIS_Z = ( 60, 140, 255)
C_TEXT   = (255, 255, 255)
C_SHADOW = (  0,   0,   0)
C_PANEL  = ( 15,  15,  15)

LW = 2


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def pt(xy):
    return (int(xy[0]), int(xy[1]))


def load_font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_dashed_line(draw, p1, p2, color, width=1, dash=8, gap=5):
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos, drawing = 0, True
    while pos < length:
        seg = dash if drawing else gap
        end = min(pos + seg, length)
        if drawing:
            draw.line([(int(x1 + ux*pos), int(y1 + uy*pos)),
                       (int(x1 + ux*end), int(y1 + uy*end))],
                      fill=color, width=width)
        pos += seg
        drawing = not drawing


def draw_arrow(draw, p_start, p_end, color, width=2, head_frac=0.22):
    x1, y1 = p_start; x2, y2 = p_end
    draw.line([pt(p_start), pt(p_end)], fill=color, width=width)
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 4:
        return
    ux, uy = dx / length, dy / length
    head = length * head_frac
    ang = math.radians(25)
    for sign in (+1, -1):
        hx = x2 - head * (ux*math.cos(ang) - sign*uy*math.sin(ang))
        hy = y2 - head * (uy*math.cos(ang) + sign*ux*math.sin(ang))
        draw.line([pt(p_end), (int(hx), int(hy))], fill=color, width=width)


def draw_cross(draw, center, radius, color, width=2):
    cx, cy = int(center[0]), int(center[1])
    draw.line([(cx-radius, cy), (cx+radius, cy)], fill=color, width=width)
    draw.line([(cx, cy-radius), (cx, cy+radius)], fill=color, width=width)
    r2 = radius // 2
    draw.ellipse([(cx-r2, cy-r2), (cx+r2, cy+r2)], fill=color)


def draw_face(draw, corners, indices, color, vis_list, width):
    n = len(indices)
    for k in range(n):
        i, j = indices[k], indices[(k+1) % n]
        vi = vis_list[i] if i < len(vis_list) else 2
        vj = vis_list[j] if j < len(vis_list) else 2
        p1, p2 = corners[i][:2], corners[j][:2]
        mv = min(vi, vj)
        if mv == 0:
            continue
        elif mv == 1:
            draw_dashed_line(draw, pt(p1), pt(p2), C_OCCL, width=max(1, width-1))
        else:
            draw.line([pt(p1), pt(p2)], fill=color, width=width)


def draw_pillar(draw, corners, i, j, vis_list):
    vi = vis_list[i] if i < len(vis_list) else 2
    vj = vis_list[j] if j < len(vis_list) else 2
    p1, p2 = corners[i][:2], corners[j][:2]
    mv = min(vi, vj)
    if mv == 0:
        return
    elif mv == 1:
        draw_dashed_line(draw, pt(p1), pt(p2), C_OCCL, width=1)
    else:
        draw.line([pt(p1), pt(p2)], fill=C_PILLAR, width=LW)


# 코너 레이아웃
#  0: rear-left-bottom   1: rear-right-bottom
#  2: rear-right-top     3: rear-left-top
#  4: front-left-bottom  5: front-right-bottom
#  6: front-right-top    7: front-left-top
REAR_FACE  = [0, 1, 2, 3]
FRONT_FACE = [4, 5, 6, 7]
PILLARS    = [(0,4),(1,5),(2,6),(3,7)]


# ── 핵심 시각화 ───────────────────────────────────────────────────────────────

def visualize(image_path, label_path, output_path):
    with open(label_path, encoding='utf-8') as f:
        lbl = json.load(f)

    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    IW, IH = img.size

    gt       = lbl["ground_truth"]
    meta     = lbl.get("metadata", {})
    dims     = lbl["truck_dims"]
    corners  = gt["2d_corners"]
    center   = gt.get("truck_center_2d") or gt.get("truck_origin_2d", [IW/2, IH/2])
    yaw      = gt["yaw_theta"]
    axes     = gt.get("axes_2d")
    view_cat = lbl.get("view_category", "?")
    vis_list = [int(c[2]) if len(c) >= 3 else 2 for c in corners]

    # ── 3D Bounding Box ────────────────────────────────────────────────────
    draw_face(draw, corners, REAR_FACE,  C_REAR,  vis_list, LW+2)
    draw_face(draw, corners, FRONT_FACE, C_FRONT, vis_list, LW+2)
    for i, j in PILLARS:
        draw_pillar(draw, corners, i, j, vis_list)

    # ── 중심 마커 ──────────────────────────────────────────────────────────
    draw_cross(draw, center, 12, C_CENTER, width=3)

    # ── XYZ 축 화살표 ─────────────────────────────────────────────────────
    font_ax = load_font(17)
    if axes:
        orig = axes["origin"]
        for lbl_str, key, color in [("X", "x_end", C_AXIS_X),
                                     ("Y", "y_end", C_AXIS_Y),
                                     ("Z", "z_end", C_AXIS_Z)]:
            end = axes[key]
            draw_arrow(draw, orig, end, color, width=3)
            ex, ey = int(end[0])+4, int(end[1])-18
            draw.text((ex+1, ey+1), lbl_str, fill=(*C_SHADOW, 220), font=font_ax)
            draw.text((ex,   ey),   lbl_str, fill=color,             font=font_ax)
    else:
        # 구버전: yaw 방향 화살표만
        rad = math.radians(yaw)
        arrow_len = min(IW, IH) * 0.07
        ox, oy = center[0], center[1]
        draw_arrow(draw, (ox, oy),
                   (ox + arrow_len*math.sin(rad), oy - arrow_len*math.cos(rad)),
                   C_AXIS_Y, width=3)

    # ── 정보 패널 (우측) ───────────────────────────────────────────────────
    font_lg = load_font(22)
    font_md = load_font(18)
    font_sm = load_font(15)

    frame_id = lbl.get("frame_id", "?")
    dist     = meta.get("distance")
    cam_pos  = meta.get("cam_pos")
    h_cam    = meta.get("h_cam")
    ty_world = meta.get("truck_yaw_world")
    vc       = {0: vis_list.count(0), 1: vis_list.count(1), 2: vis_list.count(2)}

    dist_str = f"{dist:.2f} m"     if dist     is not None else "N/A"
    h_str    = f"{h_cam:.2f} m"    if h_cam    is not None else "N/A"
    tyw_str  = f"{ty_world:.1f}°"  if ty_world is not None else "N/A"
    cp_str   = (f"({cam_pos[0]:+.1f}, {cam_pos[1]:+.1f}, {cam_pos[2]:.1f})"
                if cam_pos else "N/A")

    rows = [
        # (text, font, color)
        ("── Frame ─────────────────────", font_md, C_CENTER),
        (f"ID:          {frame_id}",       font_md, C_TEXT),
        (f"View:        {view_cat}",        font_md, C_TEXT),
        ("",                                font_sm, C_TEXT),
        ("── Truck ─────────────────────", font_md, C_CENTER),
        (f"W {dims['width']:.2f}m  L {dims['length']:.2f}m  H {dims['height']:.2f}m",
                                            font_sm, C_TEXT),
        (f"Yaw (cam):   {yaw:.1f}°",        font_md, C_TEXT),
        (f"Yaw (world): {tyw_str}",          font_md, C_TEXT),
        ("",                                font_sm, C_TEXT),
        ("── Camera ────────────────────", font_md, C_CENTER),
        (f"Distance:    {dist_str}",         font_md, C_TEXT),
        (f"Height:      {h_str}",            font_md, C_TEXT),
        (f"Pos: {cp_str}",                   font_sm, C_TEXT),
        ("",                                font_sm, C_TEXT),
        ("── Keypoints ─────────────────", font_md, C_CENTER),
        (f"Visible  (2): {vc[2]} / 8",      font_md, (100, 230, 100)),
        (f"Occluded (1): {vc[1]} / 8",      font_md, (230, 210,  80)),
        (f"Behind   (0): {vc[0]} / 8",      font_md, (220, 100, 100)),
        ("",                                font_sm, C_TEXT),
        ("── Legend ────────────────────", font_md, C_CENTER),
        ("■  Rear face",                    font_sm, C_REAR),
        ("■  Front face",                   font_sm, C_FRONT),
        ("■  X axis (right)",               font_sm, C_AXIS_X),
        ("■  Y axis (forward)",             font_sm, C_AXIS_Y),
        ("■  Z axis (up)",                  font_sm, C_AXIS_Z),
        ("- - Occluded edge",               font_sm, C_OCCL),
    ]

    LINE_H  = 24
    PANEL_W = 310
    panel_h = sum(LINE_H if t else LINE_H//2 for t, _, _ in rows) + 20
    px0 = IW - PANEL_W - 10
    py0 = max(8, IH//2 - panel_h//2)

    draw.rectangle([(px0-8, py0-8), (IW-4, py0+panel_h)],
                   fill=(*C_PANEL, 210))

    y = py0
    for text, font, color in rows:
        if text == "":
            y += LINE_H // 2
            continue
        draw.text((px0+1, y+1), text, fill=(*C_SHADOW, 200), font=font)
        draw.text((px0,   y),   text, fill=color,             font=font)
        y += LINE_H

    # 좌상단 헤더
    header = f"#{frame_id}  [{view_cat}]"
    draw.text((19, 19), header, fill=C_SHADOW, font=font_lg)
    draw.text((18, 18), header, fill=C_TEXT,   font=font_lg)

    img.save(output_path)


# ── 진입점 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1",
                        help="데이터셋 버전 (예: v1, v0_beta)")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--test", default=None,
                        help="테스트 뷰: rear | front | left | right | all")
    args = parser.parse_args()

    base       = os.path.join(SCRIPT_DIR, "datasets", args.version)
    IMAGE_DIR  = os.path.join(base, "images")
    LABEL_DIR  = os.path.join(base, "labels")
    OUTPUT_DIR = os.path.join(base, "visualized")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.test is not None:
        views = ['rear','front','left','right'] if args.test=='all' else [args.test]
        pairs = []
        for v in views:
            ip = os.path.join(IMAGE_DIR, f"test_{v}.png")
            lp = os.path.join(LABEL_DIR, f"test_{v}.json")
            if os.path.isfile(ip) and os.path.isfile(lp):
                pairs.append((f"test_{v}", ip, lp))
            else:
                print(f"  Not found: test_{v}")
    else:
        label_files = sorted(
            f for f in os.listdir(LABEL_DIR)
            if f.startswith("label_") and f.endswith(".json")
        )
        if args.idx is not None:
            label_files = [f"label_{args.idx:04d}.json"]
        elif args.num is not None:
            label_files = label_files[:args.num]
        pairs = []
        for lf in label_files:
            idx = lf.replace("label_","").replace(".json","")
            ip  = os.path.join(IMAGE_DIR, f"image_{idx}.png")
            if os.path.isfile(ip):
                pairs.append((idx, ip, os.path.join(LABEL_DIR, lf)))

    print(f"Visualizing {len(pairs)} image(s)  [{args.version}] -> {OUTPUT_DIR}")
    for name, ip, lp in pairs:
        op = os.path.join(OUTPUT_DIR, f"viz_{name}.png")
        visualize(ip, lp, op)
        print(f"  [{name}] -> {op}")
    print("Done.")


if __name__ == "__main__":
    main()
