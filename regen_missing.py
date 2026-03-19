"""
regen_missing.py
=================
레이블은 있지만 이미지가 없는 샘플(1921, 3386)을 재렌더링합니다.
레이블의 cam_pos / truck_yaw_world 값을 그대로 복원하여 동일한 장면을 재현합니다.

실행:
    blender --background --python regen_missing.py
"""

import sys, os, json, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# generate_synthetic_dataset.py의 설정값과 함수를 재사용하기 위해
# main() 실행을 막고 exec로 로드
src = open(os.path.join(SCRIPT_DIR, 'generate_synthetic_dataset.py')).read()
src = src.replace(
    "if __name__ == '__main__':",
    "if __name__ == '__regen_never__':"
)

globs = {
    '__name__': '__regen__',
    '__file__': os.path.join(SCRIPT_DIR, 'generate_synthetic_dataset.py'),
}
exec(compile(src, 'generate_synthetic_dataset.py', 'exec'), globs)

# ── 필요한 함수/변수 참조 ──────────────────────────────────────────────────────
import bpy, mathutils

clear_scene             = globs['clear_scene']
setup_render_settings   = globs['setup_render_settings']
import_truck            = globs['import_truck']
import_environment_map  = globs['import_environment_map']
create_camera           = globs['create_camera']
setup_interior_lighting = globs['setup_interior_lighting']
set_camera_look_at      = globs['set_camera_look_at']
_render_depth_pass      = globs['_render_depth_pass']
_exr_to_npy             = globs['_exr_to_npy']
MAP_CONFIGS             = globs['MAP_CONFIGS']
IMAGE_DIR               = globs['IMAGE_DIR']
DEPTH_DIR               = globs['DEPTH_DIR']
LABEL_DIR               = globs['LABEL_DIR']

# ── 재렌더링 대상 ─────────────────────────────────────────────────────────────
MISSING_INDICES = [1921, 3386]

# ── 씬 초기화 ─────────────────────────────────────────────────────────────────
clear_scene()
setup_render_settings()

truck_empty, truck_dims = import_truck()

# 두 샘플 모두 warehouse 환경
import_environment_map('warehouse')
setup_interior_lighting()

cam_obj = create_camera()
scene   = bpy.context.scene

for idx in MISSING_INDICES:
    img_path = os.path.join(IMAGE_DIR, f"image_{idx:04d}.png")
    if os.path.exists(img_path):
        print(f"[skip] image_{idx:04d}.png 이미 존재")
        continue

    # 레이블에서 파라미터 복원
    lb   = json.loads(open(os.path.join(LABEL_DIR, f"label_{idx:04d}.json")).read())
    meta = lb['metadata']

    cam_pos_xyz   = meta['cam_pos']
    truck_yaw_deg = meta['truck_yaw_world']
    truck_yaw_rad = math.radians(truck_yaw_deg)

    # 트럭 회전
    truck_empty.rotation_euler = (0.0, 0.0, truck_yaw_rad)

    # 카메라 위치 + 방향
    cam_obj.location = mathutils.Vector(cam_pos_xyz)
    set_camera_look_at(cam_obj, mathutils.Vector(truck_empty.location))

    bpy.context.view_layer.update()

    # RGB 렌더링
    scene.render.filepath = os.path.abspath(img_path)
    bpy.ops.render.render(write_still=True)
    print(f"[done] image_{idx:04d}.png")

    # Depth 렌더링 (.npy 이미 있으면 건너뜀)
    npy_path = os.path.join(DEPTH_DIR, f"depth_{idx:04d}.npy")
    if not os.path.exists(npy_path):
        _render_depth_pass(scene, DEPTH_DIR, idx, cam_obj)
        exr_path = os.path.join(DEPTH_DIR, f"depth_{idx:04d}.exr")
        png_path = os.path.join(DEPTH_DIR, f"depth_{idx:04d}.png")
        if os.path.exists(exr_path):
            _exr_to_npy(exr_path, npy_path, png_path)
        print(f"[done] depth_{idx:04d}.npy")
    else:
        print(f"[skip] depth_{idx:04d}.npy 이미 존재")

print("\n완료. 이제 split.json을 갱신하세요:")
print("  python3 -c \"from train.ablation_study import prepare_split; prepare_split(force=True)\"")
