#!/usr/bin/env python3
"""
generate_synthetic_dataset.py
==============================
현대 포터 트럭 FBX 모델 기반 합성 데이터셋 자동 생성 스크립트

실행 방법:
    blender --background --python generate_synthetic_dataset.py

생성 결과:
    datasets/v1/
    ├── images/   image_0000.png ...
    └── labels/   label_0000.json ...

JSON 레이블 스키마:
    {
      "frame_id": 0,
      "truck_dims": {"width": w, "length": l, "height": h},   # 실제 모델 제원 (m)
      "metadata": {
        "h_cam":      <float>,        # 카메라 Z 높이 (m, 월드 기준)
        "cam_pos":    [x, y, z],      # 카메라 월드 좌표 (m)
        "distance":   <float>,        # 카메라 ↔ 트럭 정중앙 거리 (m)
        "truck_yaw_world": <float>,   # 트럭 월드 Yaw (0~360°, Z축 회전)
        "K_matrix":   [[...], ...]    # 3×3 Intrinsic Matrix (픽셀 단위)
      },
      "ground_truth": {
        "truck_center_2d": [u, v],    # 트럭 기하 중앙의 2D 픽셀 좌표
        "yaw_theta": <float>,         # 카메라 좌표계 기준 Yaw (0~360°)
        "3d_corners": [[x,y,z]×8],   # 8개 꼭짓점 3D 월드 좌표 (AABB)
        "2d_corners": [[u,v]×8]      # 8개 꼭짓점 2D 픽셀 좌표
      }
    }

  좌표계 (truck_empty 로컬 기준):
      X : [-W/2, +W/2]   (폭, 좌우 대칭)
      Y : [-L/2, +L/2]   (길이, 뒷면=−L/2, 앞면=+L/2)
      Z : [-H/2, +H/2]   (높이, 바닥=−H/2, 지붕=+H/2)
      truck_empty 월드 위치: (0, 0, H/2)  → 트럭 바닥이 Z=0(지면)에 닿음
"""

import bpy
import bpy_extras.object_utils
import bmesh
import mathutils
import math
import json
import os
import random
import zipfile
import sys
import inspect
import textwrap

# =============================================================================
# 설정값 (Configuration)
# =============================================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

DEFAULT_DATASET_VERSION = "v3"
DATASET_VERSION = DEFAULT_DATASET_VERSION
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets", DATASET_VERSION)
IMAGE_DIR  = os.path.join(OUTPUT_DIR, "images")
DEPTH_DIR  = os.path.join(OUTPUT_DIR, "depth")
LABEL_DIR  = os.path.join(OUTPUT_DIR, "labels")

# FBX 모델 경로
FBX_ZIP_PATH = os.path.join(SCRIPT_DIR, "hyundai-porter-truck", "source", "포터베이크.zip")
TEXTURE_DIR  = os.path.join(SCRIPT_DIR, "hyundai-porter-truck", "textures")

# 3D 환경 맵 경로
MAP_DIR = os.path.join(SCRIPT_DIR, "map")

# 지원 맵 설정: name → {fbx, tex_dir, center_offset, scale, interior_lighting}
MAP_CONFIGS = {
    'warehouse': {
        'fbx':     os.path.join(MAP_DIR, 'warehouse-fbx-model-free', 'source', 'Warehouse.fbx'),
        'tex_dir': os.path.join(MAP_DIR, 'warehouse-fbx-model-free', 'textures'),
        # inspect 결과: AABB center X=8.0, Y=23.09 → 트럭 원점(0,0) 기준 정렬
        'center_offset': (-8.0, -23.09, 0.0),
        'scale':   1.0,
        'interior': True,   # 실내 → HDRI 비활성, 실내 조명 사용
    },
    'city': {
        'fbx':     os.path.join(MAP_DIR, 'modern-city-block', 'source', 'city_block.fbx'),
        'tex_dir': os.path.join(MAP_DIR, 'modern-city-block', 'textures'),
        # inspect 결과: 도로 중심 X≈-100, Y≈-158, 도로 최상단 Z=-0.08 → +0.08 올려 Z=0 정렬
        'center_offset': (100.0, 158.0, 0.08),
        'scale':   1.0,
        'interior': False,  # 실외 → HDRI + Sun 조명 사용
    },
    'takamatsu': {
        'fbx':     os.path.join(MAP_DIR, 'takamatsu-city', 'source', 'takamatsu_japan_sample.fbx'),
        'tex_dir': os.path.join(MAP_DIR, 'takamatsu-city', 'source'),
        # inspect 결과: 중심이 원점에 가깝고 최저면 Z≈0
        'center_offset': (0.0, 0.0, 0.0),
        # 원본 샘플이 miniature 규모(폭≈3.5m)로 들어와서 실제 도심 블록 크기에 가깝게 확장
        'scale':   40.0,
        'interior': False,  # 실외 → HDRI + Sun 조명 사용
    },
    'cnr_middle_nowhere': {
        'fbx':     os.path.join(MAP_DIR, 'cnr-ds-middle-of-nowhere', 'source', 'untitled.fbx'),
        'tex_dir': os.path.join(MAP_DIR, 'cnr-ds-middle-of-nowhere', 'source'),
        # 프리뷰 스크립트에서 실제 AABB 기준으로 재정렬한다.
        'center_offset': (0.0, 0.0, 0.0),
        'scale':   1.0,
        'interior': False,
    },
}

# ── FBX 방향 보정 옵션 ────────────────────────────────────────────────────────
# 임포트 후 모델의 길이 축이 X 방향이면 True로 설정 → Z축 기준 90° 자동 회전
ROTATE_90_DEG_IF_X_IS_LONGEST = True
# 트럭의 "뒷면(카메라 접근 방향)"이 +Y 쪽이면 True → Y축 반전 적용
# False: min_y = 뒷면(rear), True: max_y = 뒷면
FLIP_TRUCK_Y_AXIS = False

# 렌더링 파라미터
# blender --background --python generate_synthetic_dataset.py -- --total 2000
# 인자 없으면 기본값 1000
def _parse_num_images(default=1000) -> int:
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--total', type=int, default=default)
    # Blender는 '--' 뒤의 인자를 스크립트에 넘김
    try:
        idx = sys.argv.index('--')
        args, _ = parser.parse_known_args(sys.argv[idx + 1:])
    except ValueError:
        args = parser.parse_args([])
    return args.total

NUM_IMAGES    = _parse_num_images(default=1000)
RENDER_WIDTH  = 1920
RENDER_HEIGHT = 1080

# 카메라 고정 파라미터
FOCAL_LENGTH_MM = 35.0   # 초점 거리 (mm)
SENSOR_WIDTH_MM = 36.0   # 센서 가로 (mm, 35mm 풀프레임)

# 카메라 고도 범위 (모든 뷰 공통)
CAM_Z_MIN, CAM_Z_MAX = 0.5, 2.0   # 지면 ~ 2m (리프터 상하 운동)

# 카메라-트럭 거리 범위 (후면·전면·측면 공통)
CAM_DIST_MIN = 1.0    # 최소 거리 (m)
CAM_DIST_MAX = 10.0   # 최대 거리 (m)

# 카메라 후보 재시도 / 가시성 필터
MAX_CAMERA_SAMPLING_ATTEMPTS = 40
MIN_TRUCK_CLEAR_RATIO = 0.50
MIN_RENDER_MEAN_LUMA = 5.0
MIN_RENDER_MAX_CHANNEL = 20

# Look-at 노이즈 (Truncation 모사)
# 거리에 비례하여 노이즈를 스케일합니다.
#   실제 노이즈 = 카메라-목표 거리 × LOOKAT_NOISE_RATIO
#   예) 거리 2m → ±0.2m,  거리 15m → ±1.5m  (ratio=0.1)
# 0.0 = 항상 트럭 정중앙,  0.2 = 거리의 20%까지 이탈 허용
LOOKAT_NOISE_RATIO = 0.2

# ── 뷰 카테고리 가중치 (합=1.0) ──────────────────────────────────────────────
VIEW_WEIGHTS = {
    'rear':  0.25,
    'front': 0.25,
    'left':  0.25,
    'right': 0.25,
}

# ── 환경 설정 (데이터셋 생성 시 배치로 순환) ──────────────────────────────────
# weight 합이 1.0이 되도록 설정. map_name=None 이면 HDRI 모드.
# city 포지션은 map_offset 으로 구분 (center_offset 덮어씀).
ENV_CONFIGS = [
    {
        'name':        'warehouse',
        'map_name':    'warehouse',
        'map_offset':  None,                    # MAP_CONFIGS['warehouse']['center_offset'] 사용
        'weight':      0.50,                    # 50장
        # 창고 폭 16m(±8m): front/rear 뷰에서 yaw=90° 시 world_x = L/2+dist
        # L/2=2.55 → dist ≤ 4.0 이면 최대 world 반경 7.12m < 8m (0.88m 여유)
        'cam_dist_max': 4.0,
    },
    {
        'name':        'city_A',
        'map_name':    'city',
        'map_offset':  (140.0, 180.0, 0.00),   # 맵 좌표 (-140,-180), 도로Z=0
        'weight':      0.25,                    # 25장
        'cam_dist_max': 10.0,                   # 실외: 기본값 유지
    },
    {
        'name':        'city_B',
        'map_name':    'city',
        'map_offset':  (50.0, 190.0, 1.489),   # 맵 좌표 (-50,-190), 도로Z=-1.489
        'weight':      0.25,                    # 25장
        'cam_dist_max': 10.0,
        # city_B 주변 지형이 z_offset(+1.489m)으로 상승 → 카메라가 1.6m 미만이면
        # 수평 시선이 지형에 가로막혀 트럭이 보이지 않는 현상 발생
        'cam_z_min':   1.6,
    },
]

TAKAMATSU_ENV_CONFIG = {
    'name':         'takamatsu_openfield',
    'map_name':     'takamatsu',
    'map_offset':   None,
    'weight':       1.0,
    'cam_dist_max': 16.0,
    'cam_height_above_ground_min': 0.3,
    'cam_height_above_ground_max': 1.8,
    'truck_spawn_xy': (15.709855511784554, -35.311138013693004),
    'truck_spawn_clearance_m': 0.03,
}

CNR_MIDDLE_NOWHERE_ENV_CONFIG = {
    'name':         'cnr_middle_nowhere',
    'map_name':     'cnr_middle_nowhere',
    'map_offset':   None,
    'weight':       1.0,
    'cam_dist_max': 16.0,
    'cam_height_above_ground_min': 0.2,
    'cam_height_above_ground_max': 1.8,
    # 프리뷰에서 실제 dirt 면 위로 확인한 스폰 위치
    'truck_spawn_xy': (3.2690, 3.7333),
    'truck_spawn_clearance_m': 0.03,
}

# 임포트된 트럭 메쉬 오브젝트 목록 (GT 계산용 전역 참조)
TRUCK_MESH_OBJECTS: list = []
# 실제 모델에서 계산된 트럭 제원 (import_truck() 실행 후 채워짐)
TRUCK_DIMS: dict = {"width": 0.0, "length": 0.0, "height": 0.0}


def configure_output_paths(dataset_version: str) -> None:
    """Update dataset output directories at runtime."""
    global DATASET_VERSION, OUTPUT_DIR, IMAGE_DIR, DEPTH_DIR, LABEL_DIR
    DATASET_VERSION = dataset_version.strip()
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets", DATASET_VERSION)
    IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")
    LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")


def _first_missing_index(existing_indices: list[int]) -> int:
    """0부터 시작하는 frame_id 중 비어 있는 첫 인덱스를 반환합니다."""
    used = set(existing_indices)
    idx = 0
    while idx in used:
        idx += 1
    return idx

# ── 재질명 → 텍스처 파일명 수동 매핑 ─────────────────────────────────────────
# FBX가 잘못된 경로를 저장하거나 이미지명이 실제 파일명과 달라
# 자동 감지가 실패하는 재질에 대해 명시적으로 지정합니다.
# 파일명은 TEXTURE_DIR 기준 상대 경로입니다.
MATERIAL_TEXTURE_MAP: dict = {
    '매테리얼':      'internal_ground_ao_texture.jpeg',  # 바닥 그림자 AO
    '번호판이미지':   'QQQ.jpeg',                          # 번호판 이미지
    '전조등사진':    'XZCZ.png',                           # 전조등 사진
    ' 후미등이미지':  'B-removebg-preview.png',            # 후미등 이미지 (FBX 재질명 앞 공백 포함)
    '후미등이미지':   'B-removebg-preview.png',            # 후미등 이미지 (공백 없는 경우 대비)
}


# =============================================================================
# 씬 초기화
# =============================================================================

def clear_scene():
    """모든 오브젝트 및 고아 데이터 블록을 제거합니다."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def _is_descendant_of(obj: bpy.types.Object | None, parent: bpy.types.Object) -> bool:
    current = obj
    while current is not None:
        if current == parent:
            return True
        current = current.parent
    return False


def _ground_height_at(scene: bpy.types.Scene, truck_empty, x: float, y: float) -> float | None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    direction = mathutils.Vector((0.0, 0.0, -1.0))
    origin = mathutils.Vector((x, y, 500.0))

    ignored_surface_keywords = (
        'sky', 'corn', 'hay', 'fence', 'tree', 'rock',
        'windmill', 'house', 'goal', 'window', 'door',
    )

    for _ in range(16):
        hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, origin, direction)
        if not hit or _is_descendant_of(hit_obj, truck_empty):
            return None

        texts = [hit_obj.name.lower()]
        for slot in getattr(hit_obj, 'material_slots', []):
            mat = getattr(slot, 'material', None)
            if mat is not None:
                texts.append(mat.name.lower())

        if any(keyword in text for keyword in ignored_surface_keywords for text in texts):
            origin = location + direction * 0.05
            continue
        return float(location.z)
    return None


def apply_env_truck_pose(scene, truck_empty, truck_dims: dict, env_cfg: dict) -> None:
    """Optionally place the truck at an environment-specific spawn point."""
    spawn_xy = env_cfg.get('truck_spawn_xy')
    if not spawn_xy:
        return

    ground_z = _ground_height_at(scene, truck_empty, float(spawn_xy[0]), float(spawn_xy[1]))
    if ground_z is None:
        print(f"  [env] ground hit failed for spawn {spawn_xy}, keeping default truck pose")
        return

    clearance = float(env_cfg.get('truck_spawn_clearance_m', 0.0))
    truck_empty.location = (
        float(spawn_xy[0]),
        float(spawn_xy[1]),
        ground_z + (truck_dims['height'] / 2.0) + clearance,
    )
    if 'truck_yaw_world_fixed' in env_cfg:
        truck_empty.rotation_euler = (
            0.0,
            0.0,
            math.radians(float(env_cfg['truck_yaw_world_fixed'])),
        )
    bpy.context.view_layer.update()


def _ray_hits_truck_before_occluder(
    scene: bpy.types.Scene,
    origin: mathutils.Vector,
    target: mathutils.Vector,
    truck_empty,
    origin_offset_m: float = 0.05,
    hit_margin_m: float = 0.05,
) -> bool:
    """카메라→타깃 선분 상에서 트럭이 먼저 맞는지 검사합니다."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    delta = target - origin
    dist = delta.length
    if dist <= 1e-6:
        return False

    direction = delta.normalized()
    ray_origin = origin + direction * min(origin_offset_m, dist * 0.25)
    ray_dist = max(hit_margin_m, dist + hit_margin_m)

    hit, location, _, _, hit_obj, _ = scene.ray_cast(
        depsgraph, ray_origin, direction, distance=ray_dist
    )
    if not hit:
        return False

    hit_dist = (location - ray_origin).length
    if hit_dist > ray_dist:
        return False
    return _is_descendant_of(hit_obj, truck_empty)


def _get_truck_visibility_sample_points(
    truck_dims: dict,
    truck_empty,
    truck_corners_world: list | None = None,
) -> list[mathutils.Vector]:
    """환경 가림 여부를 판단할 대표 샘플 포인트들을 반환합니다."""
    center_world = mathutils.Vector(truck_empty.location)
    points = [center_world]

    W, L, H = truck_dims['width'], truck_dims['length'], truck_dims['height']
    rot_mat = mathutils.Matrix.Rotation(float(truck_empty.rotation_euler.z), 3, 'Z')
    local_points = [
        mathutils.Vector((0.0, +L / 2.0, 0.0)),          # front center
        mathutils.Vector((0.0, -L / 2.0, 0.0)),          # rear center
        mathutils.Vector((+W / 2.0, 0.0, 0.0)),          # right center
        mathutils.Vector((-W / 2.0, 0.0, 0.0)),          # left center
        mathutils.Vector((0.0, 0.0, +H * 0.45)),         # roof-ish center
    ]
    points.extend(center_world + (rot_mat @ p) for p in local_points)
    return points


def _evaluate_truck_view_clearance(
    scene: bpy.types.Scene,
    cam_obj,
    truck_empty,
    truck_dims: dict,
    truck_corners_world: list | None = None,
) -> dict:
    """카메라에서 트럭으로 가는 시선이 환경에 막히지 않는지 평가합니다."""
    sample_points = _get_truck_visibility_sample_points(
        truck_dims, truck_empty, truck_corners_world
    )
    visible_flags = [
        _ray_hits_truck_before_occluder(
            scene,
            mathutils.Vector(cam_obj.location),
            point,
            truck_empty,
        )
        for point in sample_points
    ]
    visible_count = sum(1 for flag in visible_flags if flag)
    sample_count = len(visible_flags)
    clear_ratio = (visible_count / sample_count) if sample_count else 0.0
    return {
        "center_visible": bool(visible_flags[0]) if visible_flags else False,
        "visible_count": visible_count,
        "sample_count": sample_count,
        "clear_ratio": clear_ratio,
    }


def _rendered_image_is_usable(image_path: str) -> tuple[bool, dict]:
    """
    거의 검은 프레임처럼 명백히 무효인 렌더를 걸러냅니다.
    Pillow가 없으면 이 체크는 통과 처리합니다.
    """
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return True, {"mean_luma": None, "max_channel": None}

    with Image.open(image_path) as img:
        rgb = img.convert('RGB')
        mean_luma = float(sum(ImageStat.Stat(rgb).mean) / 3.0)
        max_channel = max(ch_max for _, ch_max in rgb.getextrema())

    ok = (
        mean_luma >= MIN_RENDER_MEAN_LUMA and
        max_channel >= MIN_RENDER_MAX_CHANNEL
    )
    return ok, {
        "mean_luma": mean_luma,
        "max_channel": max_channel,
    }


def _resolve_camera_height_limits(
    truck_empty,
    truck_dims: dict,
    env_cfg: dict,
) -> tuple[float, float]:
    """
    환경 설정에서 카메라 높이 범위를 sample_camera_pose용 높이 값으로 계산합니다.
    `cam_height_above_ground_*`는 실제 지면 기준 높이로 해석하고,
    sample_camera_pose 내부의 clearance 중복을 상쇄하기 위해 spawn clearance를 보정합니다.
    """
    if (
        'cam_height_above_ground_min' in env_cfg or
        'cam_height_above_ground_max' in env_cfg
    ):
        clearance = float(env_cfg.get('truck_spawn_clearance_m', 0.0))
        z_min = max(
            0.0,
            float(env_cfg.get('cam_height_above_ground_min', CAM_Z_MIN)) - clearance,
        )
        z_max = max(
            0.0,
            float(env_cfg.get('cam_height_above_ground_max', CAM_Z_MAX)) - clearance,
        )
    else:
        z_min = float(env_cfg.get('cam_z_min', CAM_Z_MIN))
        z_max = float(env_cfg.get('cam_z_max', CAM_Z_MAX))

    if z_min > z_max:
        z_min, z_max = z_max, z_min
    return z_min, z_max


def setup_render_settings():
    scene = bpy.context.scene

    # ── 렌더러: Cycles ────────────────────────────────────────────────────
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles

    # ── 장치 설정: CPU + GPU 동시 사용 (M4 Pro Unified Memory 최적화) ────
    # M4 Pro는 CPU(12코어)와 GPU(20코어)가 동일 메모리를 공유하므로
    # 두 장치를 동시에 활성화하면 32개 처리 단위가 병렬 렌더링.
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        gpu_types = ['METAL', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI']
        found_gpu = False
        for gpu_type in gpu_types:
            try:
                prefs.compute_device_type = gpu_type
                prefs.refresh_devices()
                devices = prefs.get_devices_for_type(gpu_type)
                if devices:
                    # GPU + CPU 모두 활성화 (Unified Memory 환경에서 최대 성능)
                    active = []
                    for d in devices:
                        d.use = True
                        active.append(f"{d.name}({d.type})")
                    cycles.device = 'GPU'
                    found_gpu = True
                    break
            except Exception:
                continue
        if not found_gpu:
            cycles.device = 'CPU'
    except Exception:
        cycles.device = 'CPU'

    # ── 샘플 & 노이즈 제거 ───────────────────────────────────────────────
    cycles.samples           = 256
    cycles.use_denoising     = True
    cycles.denoiser          = 'OPENIMAGEDENOISE'
    # 1샘플부터 노이즈 제거 시작: 낮은 샘플로도 깨끗한 결과 유지
    try:
        cycles.denoising_start_sample = 1
    except AttributeError:
        pass

    # ── Persistent Data (가장 큰 속도 향상) ─────────────────────────────
    # 트럭 메쉬는 100장 내내 동일 → BVH/텍스처를 GPU 메모리에 유지
    # 매 프레임마다 씬을 재업로드하지 않아 2~4배 빠름
    scene.render.use_persistent_data = True

    # ── Camera / Distance Culling ────────────────────────────────────────
    # 카메라 뷰 밖 / 일정 거리 밖 오브젝트를 렌더에서 제외
    try:
        cycles.use_camera_cull    = True
        cycles.camera_cull_margin = 0.1   # 카메라 frustum 10% 여유
        cycles.use_distance_cull  = True
        cycles.distance_cull_margin = 50.0  # 50m 이상 오브젝트 컬링
    except AttributeError:
        pass

    scene.render.resolution_x          = RENDER_WIDTH
    scene.render.resolution_y          = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGB'
    scene.render.film_transparent = False


# =============================================================================
# Blender 5.0 FBX 임포터 버그 패치
# =============================================================================

def patch_fbx_light_bug():
    """
    Blender 5.0에서 FBX 임포터가 삭제된 Cycles API를 참조하는 버그를 패치합니다.

    증상:
        AttributeError: 'CyclesLightSettings' object has no attribute 'cast_shadow'

    원인:
        Blender 5.0에서 Cycles 라이트의 cast_shadow 속성이 제거되었으나,
        내장 FBX 임포터(import_fbx.py)가 아직 이를 참조하고 있습니다.

    해결 방법:
        1. io_scene_fbx.import_fbx 모듈을 로드
        2. inspect.getsource()로 blen_read_light 함수 소스 추출
        3. 문제 라인을 hasattr() 가드로 교체
        4. exec()로 재컴파일 후 모듈에 재주입
    """
    # ── FBX 애드온 경로를 sys.path에 추가 ──────────────────────────────────
    # blender 실행 파일 위치: .app/Contents/MacOS/blender
    # 리소스 위치:            .app/Contents/Resources/<version>/scripts/addons_core
    blender_bin = bpy.app.binary_path
    macos_dir   = os.path.dirname(blender_bin)                         # .../MacOS
    resources   = os.path.normpath(os.path.join(macos_dir, '..', 'Resources'))
    ver_str     = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
    addons_core = os.path.join(resources, ver_str, 'scripts', 'addons_core')

    if addons_core not in sys.path:
        sys.path.insert(0, addons_core)

    # ── 모듈 로드 ────────────────────────────────────────────────────────────
    try:
        import io_scene_fbx.import_fbx as fbx_mod
    except ImportError:
        return

    # ── 패치 필요 여부 확인 ──────────────────────────────────────────────────
    TARGET_LINE  = 'lamp.cycles.cast_shadow = lamp.use_shadow'
    PATCHED_LINE = ('if hasattr(lamp.cycles, "cast_shadow"): '
                    'lamp.cycles.cast_shadow = lamp.use_shadow')

    try:
        src = inspect.getsource(fbx_mod.blen_read_light)
    except Exception:
        return

    if TARGET_LINE not in src:
        return

    # ── 소스 패치 → exec로 재컴파일 → 모듈에 재주입 ─────────────────────────
    patched_src = textwrap.dedent(src).replace(TARGET_LINE, PATCHED_LINE)

    # exec 네임스페이스: 모듈의 모든 전역 이름 + 빌트인 제공
    ns = {**vars(fbx_mod), '__builtins__': __builtins__}
    exec(compile(patched_src, '<fbx_cast_shadow_patch>', 'exec'), ns)

    # 패치된 함수로 교체
    fbx_mod.blen_read_light = ns['blen_read_light']


# =============================================================================
# FBX 트럭 임포트 (핵심 함수)
# =============================================================================

def import_truck() -> tuple:
    """
    ZIP에서 FBX를 추출하여 임포트하고, 원점을 "뒷면 중심 바닥"으로 정렬합니다.

    처리 흐름:
        1. ZIP 압축 해제 → 포터베이크.fbx 추출
        2. bpy.ops.import_scene.fbx() 호출
        3. 임포트된 오브젝트의 부모(Parent) 관계 해제 (keep transform)
        4. 모든 트랜스폼 Apply (location/rotation/scale → vertex data로 굽기)
        5. 전체 메쉬의 월드 AABB(Axis-Aligned Bounding Box) 계산
        6. 단위 자동 보정 (FBX가 cm 단위로 내보내진 경우 ÷100)
        7. 길이 축 정렬 — 가장 긴 축이 X이면 Z축 기준 90° 회전하여 Y축으로 정렬
        8. 뒷면(Y=min) 기준 원점 이동 → rear-center-bottom = (0,0,0)
        9. 기준 Empty 오브젝트 'Truck' 생성 (원점 표시용)
       10. 전역 변수 TRUCK_MESH_OBJECTS, TRUCK_DIMS 갱신

    Returns:
        truck_empty (bpy.types.Object): 원점에 위치한 기준 Empty
        dims        (dict)            : {"width": w, "length": l, "height": h} (m)
    """
    global TRUCK_MESH_OBJECTS, TRUCK_DIMS

    # ── Step 1: ZIP 압축 해제 ──────────────────────────────────────────────
    fbx_path = os.path.join(os.path.dirname(FBX_ZIP_PATH), "포터베이크.fbx")
    if not os.path.exists(fbx_path):
        with zipfile.ZipFile(FBX_ZIP_PATH, 'r') as zf:
            zf.extractall(os.path.dirname(FBX_ZIP_PATH))

    # ── Step 2: FBX 임포트 ────────────────────────────────────────────────
    # Blender 5.0 호환 패치 적용 (cast_shadow AttributeError 방지)
    patch_fbx_light_bug()

    # 임포트 전 오브젝트 키셋을 저장 → 이후 차집합으로 새 오브젝트 식별
    keys_before = set(bpy.data.objects.keys())

    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        global_scale=1.0,         # 미터 단위 FBX: 1.0 / cm 단위 FBX: 0.01
        use_custom_normals=True,
        use_image_search=True,    # 텍스처를 주변 경로에서 자동 탐색
        use_anim=False,           # 애니메이션 데이터 불필요
        force_connect_children=False,
    )

    imported_all  = [bpy.data.objects[k]
                     for k in set(bpy.data.objects.keys()) - keys_before]
    mesh_objs = [o for o in imported_all if o.type == 'MESH']

    if not mesh_objs:
        raise RuntimeError("FBX에서 MESH 타입 오브젝트를 찾지 못했습니다.")

    # ── Step 3: Parent 관계 해제 (Keep Transform) ─────────────────────────
    # 부모 오브젝트가 있으면 transform_apply가 올바르게 동작하지 않으므로 먼저 해제
    bpy.ops.object.select_all(action='DESELECT')
    for o in imported_all:
        o.select_set(True)
    bpy.context.view_layer.objects.active = imported_all[0]
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # ── Step 4: 트랜스폼 Apply ────────────────────────────────────────────
    # Apply 후: obj.location=(0,0,0), rotation=(0,0,0), scale=(1,1,1)
    #           vertex.co가 직접 월드 좌표를 나타냄
    bpy.ops.object.select_all(action='DESELECT')
    for o in imported_all:
        o.select_set(True)
    bpy.context.view_layer.objects.active = imported_all[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # ── Step 5: 전체 월드 AABB 계산 ──────────────────────────────────────
    # transform_apply 후: vertex.co ≈ 월드 좌표 (matrix_world ≈ Identity)
    INF = float('inf')
    mn = mathutils.Vector(( INF,  INF,  INF))
    mx = mathutils.Vector((-INF, -INF, -INF))
    for o in mesh_objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            wv = mw @ v.co
            mn.x = min(mn.x, wv.x);  mx.x = max(mx.x, wv.x)
            mn.y = min(mn.y, wv.y);  mx.y = max(mx.y, wv.y)
            mn.z = min(mn.z, wv.z);  mx.z = max(mx.z, wv.z)

    raw_w = mx.x - mn.x
    raw_l = mx.y - mn.y
    raw_h = mx.z - mn.z
    # ── Step 6: 단위 자동 보정 ───────────────────────────────────────────
    max_dim = max(raw_w, raw_l, raw_h)
    if max_dim > 15.0:
        scale = 0.01
    elif max_dim < 0.5:
        scale = 100.0
    else:
        scale = 1.0

    if scale != 1.0:
        for o in mesh_objs:
            for v in o.data.vertices:
                v.co *= scale
            o.data.update()
        mn *= scale;  mx *= scale
        raw_w *= scale;  raw_l *= scale;  raw_h *= scale

    # ── Step 7: 길이 축 정렬 (Y축이 트럭의 전후 방향이 되도록) ───────────
    # 차량의 길이(앞뒤)가 가장 길기 때문에, 가장 긴 축을 Y로 정렬합니다.
    extents = {'x': raw_w, 'y': raw_l, 'z': raw_h}
    longest = max(extents, key=extents.get)

    if longest == 'x' and ROTATE_90_DEG_IF_X_IS_LONGEST:
        # 4×4 회전 행렬: Z축 +90° (X→Y 방향 회전)
        rot = mathutils.Matrix.Rotation(math.radians(90), 4, 'Z')
        for o in mesh_objs:
            for v in o.data.vertices:
                v.co = (rot @ v.co.to_4d()).to_3d()
            o.data.update()
        # AABB 재계산
        mn = mathutils.Vector(( INF,  INF,  INF))
        mx = mathutils.Vector((-INF, -INF, -INF))
        for o in mesh_objs:
            for v in o.data.vertices:
                wv = o.matrix_world @ v.co
                mn.x = min(mn.x, wv.x);  mx.x = max(mx.x, wv.x)
                mn.y = min(mn.y, wv.y);  mx.y = max(mx.y, wv.y)
                mn.z = min(mn.z, wv.z);  mx.z = max(mx.z, wv.z)

    truck_w = mx.x - mn.x   # 폭
    truck_l = mx.y - mn.y   # 길이 (뒤→앞)
    truck_h = mx.z - mn.z   # 높이

    # ── Step 8: 원점 이동 → 기하학적 중앙 = (0, 0, 0) in 로컬 공간 ─────
    #
    # 목표 좌표 범위 (truck_empty 로컬):
    #   X : [-truck_w/2, +truck_w/2]   (폭 중심화)
    #   Y : [-truck_l/2, +truck_l/2]   (뒷면=−L/2, 앞면=+L/2)
    #   Z : [-truck_h/2, +truck_h/2]   (바닥=−H/2, 지붕=+H/2)
    #
    # truck_empty 는 월드 (0, 0, truck_h/2) 에 배치 → 트럭 바닥이 Z=0(지면)

    # Y 반전이 필요한 경우 먼저 처리
    if FLIP_TRUCK_Y_AXIS:
        for o in mesh_objs:
            for v in o.data.vertices:
                v.co.y = -v.co.y
            o.data.update()
        # AABB 재계산 (Y 반전 후)
        mn = mathutils.Vector(( float('inf'),  float('inf'),  float('inf')))
        mx = mathutils.Vector((-float('inf'), -float('inf'), -float('inf')))
        for o in mesh_objs:
            for v in o.data.vertices:
                mn.x = min(mn.x, v.co.x);  mx.x = max(mx.x, v.co.x)
                mn.y = min(mn.y, v.co.y);  mx.y = max(mx.y, v.co.y)
                mn.z = min(mn.z, v.co.z);  mx.z = max(mx.z, v.co.z)

    # 기하학적 중심으로 오프셋
    off_x = -( mn.x + mx.x) / 2.0
    off_y = -( mn.y + mx.y) / 2.0
    off_z = -( mn.z + mx.z) / 2.0
    offset = mathutils.Vector((off_x, off_y, off_z))
    for o in mesh_objs:
        for v in o.data.vertices:
            v.co += offset
        o.data.update()

    # ── Step 9: 기준 Empty('Truck') 생성 ────────────────────────────────
    # 로컬 원점 = 기하학적 중앙 (0, 0, 0)
    # 월드 위치  = (0, 0, truck_h/2) → 바닥이 월드 Z=0 (지면) 에 닿음
    # rotation_euler[2] 를 매 프레임 변경하면 지면 위에서 제자리 회전
    truck_empty = bpy.data.objects.new('Truck', None)
    truck_empty.empty_display_type = 'ARROWS'
    truck_empty.empty_display_size = 0.4
    bpy.context.scene.collection.objects.link(truck_empty)
    truck_empty.location       = (0.0, 0.0, truck_h / 2.0)
    truck_empty.rotation_euler = (0.0, 0.0, 0.0)

    # 모든 임포트 오브젝트를 Truck Empty 아래로 그룹화
    for o in imported_all:
        o.parent = truck_empty
        o.matrix_parent_inverse = mathutils.Matrix.Identity(4)

    # ── Step 10: 전역 변수 갱신 ─────────────────────────────────────────
    TRUCK_MESH_OBJECTS = mesh_objs
    TRUCK_DIMS = {"width": truck_w, "length": truck_l, "height": truck_h}
    return truck_empty, TRUCK_DIMS


# =============================================================================
# 바닥 평면 생성
# =============================================================================

def create_ground_plane(size: float = 30.0):
    """
    Shadow Catcher 바닥면 — 자체 색상은 없고 트럭 그림자만 받아
    HDRI 배경 위에 트럭이 자연스럽게 서 있는 것처럼 보이게 합니다.
    _DR_ground_mat 는 None으로 유지 (랜덤화 대상 없음).
    """
    bm = bmesh.new()
    s = size
    verts = [bm.verts.new(c) for c in [(-s,-s,0), (s,-s,0), (s,s,0), (-s,s,0)]]
    bm.faces.new(verts)
    mesh = bpy.data.meshes.new('GroundMesh')
    bm.to_mesh(mesh); bm.free()
    obj = bpy.data.objects.new('Ground', mesh)
    bpy.context.scene.collection.objects.link(obj)
    # Cycles Shadow Catcher: 렌더 결과에서 이 면은 투명하되 그림자만 수신
    obj.is_shadow_catcher = True
    return obj


# =============================================================================
# 3D 환경 맵 임포트
# =============================================================================

def import_environment_map(map_name: str) -> list:
    """
    MAP_CONFIGS 에 정의된 3D 환경 맵 FBX를 임포트하고
    트럭 원점(0,0,0) 기준으로 중심 정렬합니다.

    반환: 임포트된 오브젝트 리스트
    """
    cfg = MAP_CONFIGS.get(map_name)
    if cfg is None:
        print(f"  [map] 알 수 없는 맵: {map_name}")
        return []
    if not os.path.isfile(cfg['fbx']):
        print(f"  [map] FBX 없음: {cfg['fbx']}")
        return []

    print(f"  [map] {map_name} 임포트 중...")
    bpy.ops.import_scene.fbx(filepath=cfg['fbx'])
    bpy.context.view_layer.update()

    scale_factor = float(cfg.get('scale', 1.0))
    ox, oy, oz = cfg['center_offset']
    imported = list(bpy.context.selected_objects)
    for obj in imported:
        obj.scale.x *= scale_factor
        obj.scale.y *= scale_factor
        obj.scale.z *= scale_factor
        obj.location.x += ox
        obj.location.y += oy
        obj.location.z += oz
    bpy.context.view_layer.update()

    # 텍스처 재연결 (맵별 처리)
    if map_name == 'warehouse':
        _reconnect_warehouse_textures(cfg['tex_dir'])
    elif map_name == 'city':
        _reconnect_city_textures(cfg['tex_dir'])

    print(f"  [map] 완료: {len(imported)}개 오브젝트")
    return imported


def _reconnect_warehouse_textures(tex_dir: str):
    """WetConcrete 재질에 PBR 텍스처를 수동 연결합니다."""
    for mat in bpy.data.materials:
        if 'WetConcrete' not in mat.name:
            continue
        mat.use_nodes = True
        tree  = mat.node_tree
        nodes = tree.nodes
        links = tree.links
        nodes.clear()

        out  = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

        def add_tex(fname, colorspace='sRGB'):
            p = os.path.join(tex_dir, fname)
            if not os.path.isfile(p):
                return None
            t = nodes.new('ShaderNodeTexImage')
            t.image = bpy.data.images.load(p)
            t.image.colorspace_settings.name = colorspace
            return t

        bc  = add_tex('WetConcrete_Base_color.png', 'sRGB')
        rgh = add_tex('WetConcrete_Roughness@channels=G.png', 'Non-Color')
        nrm = add_tex('WetConcrete_Normal_DirectX.png', 'Non-Color')

        if bc:
            links.new(bc.outputs['Color'],  bsdf.inputs['Base Color'])
        if rgh:
            links.new(rgh.outputs['Color'], bsdf.inputs['Roughness'])
        if nrm:
            nm = nodes.new('ShaderNodeNormalMap')
            # DirectX Normal → OpenGL Normal: G 채널 반전
            # Blender 4+ : SeparateColor / CombineColor
            try:
                sep  = nodes.new('ShaderNodeSeparateColor')
                comb = nodes.new('ShaderNodeCombineColor')
                r_out, g_out, b_out = 'Red', 'Green', 'Blue'
                r_in,  g_in,  b_in  = 'Red', 'Green', 'Blue'
            except Exception:
                sep  = nodes.new('ShaderNodeSeparateRGB')
                comb = nodes.new('ShaderNodeCombineRGB')
                r_out, g_out, b_out = 'R', 'G', 'B'
                r_in,  g_in,  b_in  = 'R', 'G', 'B'
            inv  = nodes.new('ShaderNodeInvert')
            links.new(nrm.outputs['Color'],  sep.inputs['Color'])
            links.new(sep.outputs[r_out],    comb.inputs[r_in])
            links.new(sep.outputs[g_out],    inv.inputs['Color'])
            links.new(inv.outputs['Color'],  comb.inputs[g_in])
            links.new(sep.outputs[b_out],    comb.inputs[b_in])
            links.new(comb.outputs['Color'], nm.inputs['Color'])
            links.new(nm.outputs['Normal'],  bsdf.inputs['Normal'])

    print(f"  [map] WetConcrete 텍스처 연결 완료")


def _reconnect_city_textures(tex_dir: str):
    """
    city_block 재질(1~7)에 PBR 텍스처를 연결합니다.
    각 재질 번호에 대해 <num>_BaseColor/Metallic/Roughness/Normal.png 를 찾아 연결.
    """
    # 재질 번호별 사용 가능 텍스처 파일 매핑
    TEX_MAP = {
        '1': {'bc': '1_BaseColor.png', 'mt': '1_Metallic.png',
              'rg': '1_Roughness.png', 'nm': '1_Normal.png'},
        '2': {'bc': '2_BaseColor.png',
              'rg': '2_Roughness.png', 'nm': '2_Normal.png'},
        '3': {'bc': '3_BaseColor.png',
              'rg': '3_Roughness.png', 'nm': '3_Normal.png'},
        '4': {'bc': '4_BaseColor.png', 'mt': '4_Metallic.png',
              'rg': '4_Roughness.png'},
        '5': {'bc': '5_BaseColor.png',
              'rg': '5_Roughness.png', 'nm': '5_Normal.png'},
        '6': {'bc': '6_BaseColor.png', 'mt': '6_Metallic.png',
              'rg': '6_Roughness.png', 'nm': '6_Normal.png'},
        '7': {'bc': '7_BaseColor.png', 'mt': '7_Metallic.png',
              'rg': '7_Roughness.png', 'nm': '7_Normal.png'},
    }

    for mat in bpy.data.materials:
        tmap = TEX_MAP.get(mat.name)
        if tmap is None:
            continue
        mat.use_nodes = True
        tree  = mat.node_tree
        nodes = tree.nodes
        links = tree.links
        nodes.clear()

        out  = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

        def add_tex(fname, colorspace='sRGB'):
            p = os.path.join(tex_dir, fname)
            if not os.path.isfile(p):
                return None
            t = nodes.new('ShaderNodeTexImage')
            t.image = bpy.data.images.load(p, check_existing=True)
            t.image.colorspace_settings.name = colorspace
            return t

        bc = add_tex(tmap['bc'], 'sRGB') if 'bc' in tmap else None
        mt = add_tex(tmap['mt'], 'Non-Color') if 'mt' in tmap else None
        rg = add_tex(tmap['rg'], 'Non-Color') if 'rg' in tmap else None
        nm = add_tex(tmap['nm'], 'Non-Color') if 'nm' in tmap else None

        if bc: links.new(bc.outputs['Color'],  bsdf.inputs['Base Color'])
        if mt: links.new(mt.outputs['Color'],  bsdf.inputs['Metallic'])
        if rg: links.new(rg.outputs['Color'],  bsdf.inputs['Roughness'])
        if nm:
            nmap = nodes.new('ShaderNodeNormalMap')
            links.new(nm.outputs['Color'], nmap.inputs['Color'])
            links.new(nmap.outputs['Normal'], bsdf.inputs['Normal'])

    print(f"  [map] city 텍스처 연결 완료")


def setup_interior_lighting():
    """
    실내 3D 맵용 조명: 어두운 배경 + 천장 Area 조명 4개.
    (HDRI, Sun 없음 — 웨어하우스 Emissive 재질과 Area 조명으로 실내 분위기 구성)
    """
    global _DR_sun_obj, _DR_bg_node, _DR_env_tex

    # 배경: 약한 ambient (완전 검정은 코너가 너무 어두움)
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    out = tree.nodes.new('ShaderNodeOutputWorld')
    bg  = tree.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value    = (0.08, 0.08, 0.10, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    tree.links.new(bg.outputs['Background'], out.inputs['Surface'])
    _DR_bg_node = bg
    _DR_env_tex = None

    # 천장 Area 조명 9개 3×3 그리드 (창고 16m×46m → 트럭 주변 ±6m 커버)
    # 에너지: 5000W / 조명, 크기: 3m — 형광등 조도 수준
    grid_positions = [
        (-5,  6), (0,  6), (5,  6),
        (-5,  0), (0,  0), (5,  0),
        (-5, -6), (0, -6), (5, -6),
    ]
    for idx, (px, py) in enumerate(grid_positions):
        light_data = bpy.data.lights.new(f'AreaLight_{idx}', type='AREA')
        light_data.energy = 5000
        light_data.size   = 3.0
        light_obj = bpy.data.objects.new(f'AreaLight_{idx}', light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = (px, py, 4.5)         # 천장 근처 (Z=4.5m)
        light_obj.rotation_euler = (math.pi, 0, 0) # 아래 방향

    _DR_sun_obj = None   # 실내: 태양광 없음


# =============================================================================
# 조명 설정
# =============================================================================

def setup_lighting():
    """
    씬 조명 초기 설정.
    - Sun 오브젝트 생성 (매 프레임 randomize_domain()에서 방향/세기 교체)
    - World 노드 그래프 구성: HDR Env Tex → Background → Output
      HDR 파일 목록이 있으면 첫 번째를 로드, 없으면 절차적 하늘 폴백.
    - 글로벌 _DR_* 에 노드 참조를 저장해 두면 이후 per-frame 교체 가능.
    """
    global _DR_env_tex, _DR_bg_node, _DR_sun_obj

    # ── Sun 오브젝트 ──────────────────────────────────────────────────────
    sun_data = bpy.data.lights.new('Sun', type='SUN')
    sun_data.energy = 3.0
    sun_data.angle  = math.radians(2)
    sun_obj = bpy.data.objects.new('Sun', sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(50), 0.0, math.radians(25))
    _DR_sun_obj = sun_obj

    # ── World 노드 그래프 ─────────────────────────────────────────────────
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    out_node = tree.nodes.new('ShaderNodeOutputWorld')
    bg_node  = tree.nodes.new('ShaderNodeBackground')
    tree.links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])
    bg_node.inputs['Strength'].default_value = 1.0
    _DR_bg_node = bg_node

    # ── HDR 환경맵 (사용 가능한 경우) ────────────────────────────────────
    first_hdr = HDRI_FILES[0] if HDRI_FILES else None
    if first_hdr and os.path.isfile(first_hdr):
        env_tex = tree.nodes.new('ShaderNodeTexEnvironment')
        env_tex.image = bpy.data.images.load(first_hdr)
        rot_node = tree.nodes.new('ShaderNodeMapping')
        coord_node = tree.nodes.new('ShaderNodeTexCoord')
        tree.links.new(coord_node.outputs['Generated'], rot_node.inputs['Vector'])
        tree.links.new(rot_node.outputs['Vector'],      env_tex.inputs['Vector'])
        tree.links.new(env_tex.outputs['Color'],        bg_node.inputs['Color'])
        _DR_env_tex = env_tex
    else:
        # 폴백: 절차적 하늘
        sky_node = tree.nodes.new('ShaderNodeTexSky')
        for sky_type in ('NISHITA', 'HOSEK_WILKIE', 'PREETHAM'):
            try:
                sky_node.sky_type = sky_type; break
            except TypeError:
                pass
        tree.links.new(sky_node.outputs['Color'], bg_node.inputs['Color'])
        _DR_env_tex = None


# =============================================================================
# Domain Randomization — HDR 환경맵 목록
# =============================================================================
# hdri/ 폴더 안의 *.hdr 파일을 모두 자동으로 수집합니다.
# 각 프레임마다 무작위로 한 개를 선택하여 반사/조명 환경을 바꿉니다.

_HDRI_DIR = os.path.join(SCRIPT_DIR, "hdri")

def _collect_hdri_files() -> list:
    if not os.path.isdir(_HDRI_DIR):
        return []
    return sorted(
        os.path.join(_HDRI_DIR, f)
        for f in os.listdir(_HDRI_DIR)
        if f.lower().endswith('.hdr') or f.lower().endswith('.exr')
    )

HDRI_FILES: list = []   # main() 시작 시 채워짐

# Domain Randomization 런타임 참조 (setup_lighting 이후 채워짐)
_DR_env_tex   = None   # ShaderNodeTexEnvironment  — HDR 교체용
_DR_bg_node   = None   # ShaderNodeBackground       — 강도 교체용
_DR_sun_obj   = None   # Sun 오브젝트               — 방향·세기 교체용
_DR_ground_mat = None  # Ground 재질               — 색상·거칠기 교체용


# =============================================================================
# 카메라 생성
# =============================================================================

def create_camera():
    """35mm/F1.8 풀프레임 동등 카메라 생성 (고정 파라미터)."""
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens         = FOCAL_LENGTH_MM
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.sensor_fit   = 'HORIZONTAL'
    cam_data.clip_start   = 0.01
    cam_data.clip_end     = 200.0
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


# =============================================================================
# 카메라 Look-at
# =============================================================================

def set_camera_look_at(cam_obj: bpy.types.Object, target: mathutils.Vector):
    """
    target 방향으로 카메라를 수평 회전합니다 (Pitch=0, Roll=0).

    look 방향의 Z 성분을 0으로 강제하여 카메라가 항상 수평(지면과 평행)으로
    유지되도록 합니다. 이 경우 핀홀 공식 Z = fy * h_cam / (v_c - cy)가
    수치적으로 정확히 성립합니다.

    Blender 카메라의 광학 축 = 로컬 −Z.
    to_track_quat('-Z', 'Y'): 방향 벡터가 로컬 −Z, 로컬 Y가 세계 +Z를 향하는
    쿼터니언 반환. 방향 벡터가 수평(dz=0)이면 로컬 Y = 세계 +Z → 완전 수평.
    """
    direction = target - cam_obj.location
    # Z 성분 제거 → 수평 방향만 사용 (Yaw 만 변함, Pitch=0 보장)
    direction_h = mathutils.Vector((direction.x, direction.y, 0.0))
    if direction_h.length < 1e-6:
        return
    rot_quat = direction_h.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()


# =============================================================================
# 카메라 Intrinsic Matrix K
# =============================================================================

def get_camera_intrinsic_matrix(scene, cam_obj) -> list:
    """
    픽셀 단위 3×3 Intrinsic Matrix K를 계산합니다.

        fx = f_mm × (W / sensor_w)
        fy = f_mm × (H / sensor_h)   sensor_h = sensor_w × (H/W) → fx == fy
        cx = W/2 + shift_x×W
        cy = H/2 − shift_y×H
    """
    cam_data = cam_obj.data
    scale = scene.render.resolution_percentage / 100.0
    W = scene.render.resolution_x * scale
    H = scene.render.resolution_y * scale
    f_mm     = cam_data.lens
    sensor_w = cam_data.sensor_width
    if cam_data.sensor_fit == 'VERTICAL':
        sensor_h = cam_data.sensor_height
        sensor_w = sensor_h * (W / H)
    else:
        sensor_h = sensor_w * (H / W)
    fx = f_mm * (W / sensor_w)
    fy = f_mm * (H / sensor_h)
    cx = (W / 2.0) + cam_data.shift_x * W
    cy = (H / 2.0) - cam_data.shift_y * H
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


# =============================================================================
# 3D → 2D 투영
# =============================================================================

def world_to_image_pixel(scene, cam_obj, world_point: mathutils.Vector) -> list:
    """3D 월드 좌표 → [u, v] 픽셀 (visibility 없음, 중심점 등 단순 투영용)."""
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, world_point)
    return [ndc.x * W, (1.0 - ndc.y) * H]


def world_to_image_kp(scene, cam_obj, world_point: mathutils.Vector) -> list:
    """
    3D 월드 좌표 → [u, v, visibility] keypoint.

    visibility:
        0 = 카메라 뒤 (ndc.z ≤ 0) — 화면에 투영 불가
        1 = 화면 밖 (truncated)    — 투영은 되지만 이미지 범위 외
        2 = 정상 가시              — 이미지 안에 있음

    bpy_extras ndc.z: 카메라로부터의 거리(양수 = 앞, 음수 = 뒤)
    """
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, world_point)
    u = ndc.x * W
    v = (1.0 - ndc.y) * H
    if ndc.z <= 0.0:
        vis = 0
    elif not (0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0):
        vis = 1
    else:
        vis = 2
    return [u, v, vis]


# =============================================================================
# 트럭 AABB 꼭짓점 (실제 메쉬 기반)
# =============================================================================

def get_truck_obb_corners(truck_dims: dict, truck_empty) -> list:
    """
    트럭 Oriented Bounding Box (OBB) 8개 꼭짓점을 월드 좌표로 반환합니다.

    AABB(세계축 정렬)와 달리 트럭의 yaw 회전을 반영한 실제 모서리 좌표입니다.
    truck_dims의 W×L×H 치수가 어떤 yaw에서도 동일하게 유지됩니다.

    꼭짓점 인덱스 (KITTI 관례):
        0: rear-left-bottom   1: rear-right-bottom
        2: rear-right-top     3: rear-left-top
        4: front-left-bottom  5: front-right-bottom
        6: front-right-top    7: front-left-top
    """
    W, L, H = truck_dims['width'], truck_dims['length'], truck_dims['height']
    yaw = truck_empty.rotation_euler.z
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    tx, ty, tz = truck_empty.location   # 항상 (0, 0, H/2)

    def corner(xl, yl, zl):
        return [
            xl * cos_y - yl * sin_y + tx,
            xl * sin_y + yl * cos_y + ty,
            zl + tz,
        ]

    hw, hl, hh = W / 2.0, L / 2.0, H / 2.0
    return [
        corner(-hw, -hl, -hh),  # 0: rear-left-bottom
        corner(+hw, -hl, -hh),  # 1: rear-right-bottom
        corner(+hw, -hl, +hh),  # 2: rear-right-top
        corner(-hw, -hl, +hh),  # 3: rear-left-top
        corner(-hw, +hl, -hh),  # 4: front-left-bottom
        corner(+hw, +hl, -hh),  # 5: front-right-bottom
        corner(+hw, +hl, +hh),  # 6: front-right-top
        corner(-hw, +hl, +hh),  # 7: front-left-top
    ]


# =============================================================================
# Yaw 각도 계산
# =============================================================================

def compute_yaw_angle(cam_obj: bpy.types.Object,
                       truck_obj: bpy.types.Object) -> float:
    """
    카메라 좌표계 기준 트럭 Yaw(θ)를 계산합니다.

    1. truck_fwd_world = R_truck × [0,1,0]ᵀ   (truck_obj 로컬 +Y = 전진 방향)
    2. truck_fwd_cam   = R_cam⁻¹ × truck_fwd_world
    3. θ = atan2(truck_fwd_cam.x, −truck_fwd_cam.z) % 360   → [0°, 360°)

    θ=0°   : 트럭 전면이 카메라를 향함 (후면 뷰)
    θ=180° : 트럭 후면이 카메라를 향함 (전면 뷰)
    θ=90°  : 트럭 오른쪽이 카메라를 향함
    """
    truck_fwd_local = mathutils.Vector((0.0, 1.0, 0.0))
    truck_fwd_world = truck_obj.matrix_world.to_3x3() @ truck_fwd_local
    truck_fwd_world.normalize()

    cam_rot_inv   = cam_obj.matrix_world.to_3x3().inverted()
    truck_fwd_cam = cam_rot_inv @ truck_fwd_world

    yaw_rad = math.atan2(truck_fwd_cam.x, -truck_fwd_cam.z)
    return math.degrees(yaw_rad) % 360.0


# =============================================================================
# 텍스처 재연결 (Texture Path Remapping)
# =============================================================================

def fix_missing_textures():
    """
    FBX 임포트 후 깨진 텍스처 경로를 TEXTURE_DIR 기준으로 재연결합니다.

    문제 원인:
        FBX 파일의 내부 텍스처 경로가 잘못된 상대 경로(예: '.')를 가리키거나
        절대 경로가 현재 환경과 맞지 않아 GPU 텍스처 로드에 실패합니다.

    해결 방법:
        1. TEXTURE_DIR 내 파일 목록을 인덱싱 (이름·확장자 없는 이름 기준)
        2. bpy.data.images를 순회하며 미로드 이미지 탐색
        3. 이미지 이름을 정규화(Blender .001 suffix 제거)하여 파일 매칭
        4. 경로 재설정 후 reload()
    """
    if not os.path.isdir(TEXTURE_DIR):
        print(f"  [Texture] 텍스처 폴더 없음, 생략: {TEXTURE_DIR}")
        return

    # ── 사용 가능한 텍스처 파일 인덱스 구축 ────────────────────────────────
    # key: 소문자 파일명(확장자 포함 & 미포함 모두) → value: 전체 경로
    available: dict = {}
    for fname in os.listdir(TEXTURE_DIR):
        full = os.path.join(TEXTURE_DIR, fname)
        if not os.path.isfile(full):
            continue
        available[fname.lower()] = full
        available[os.path.splitext(fname)[0].lower()] = full

    import re as _re

    # ── Phase 1: 깨진 Image 데이터블록 경로 재연결 ────────────────────────
    fixed = 0
    for img in bpy.data.images:
        if img.name == 'Render Result':
            continue

        # 실제 파일 존재 여부로 판단 (filepath가 '.' 이나 'source/.' 같은 디렉터리를
        # 가리키는 경우도 누락으로 처리)
        abs_path = bpy.path.abspath(img.filepath) if img.filepath else ''
        if abs_path and os.path.isfile(abs_path):
            continue  # 정상 파일 → 건너뜀

        # ── 매칭 우선순위 ─────────────────────────────────────────────────
        # 1) filepath 파일명 기준  2) img.name 기준 (Blender .001 suffix 제거)
        candidates = []
        if img.filepath:
            candidates.append(os.path.basename(abs_path).lower())

        # Blender가 중복 이미지에 '.001', '.002' 등을 붙이므로 제거
        clean_name = _re.sub(r'\.\d+$', '', img.name.lower())
        candidates.append(clean_name)
        candidates.append(os.path.splitext(clean_name)[0])

        new_path = None
        for cand in candidates:
            if cand:
                new_path = available.get(cand)
            if new_path:
                break

        if new_path:
            img.filepath = new_path
            try:
                img.reload()
                fixed += 1
            except Exception:
                pass

    # ── Phase 2: 재질의 모든 Image Texture 노드에서 깨진 것 처리 ────────────
    # Base Color 외 다른 소켓(Alpha, Emission 등)에 연결된 경우도 포함합니다.
    # 우선순위:
    #   1) MATERIAL_TEXTURE_MAP에 매핑된 재질 → 지정 텍스처 파일 로드 (이미지만 교체)
    #   2) 타이어/휠 재질의 Base Color 노드 → 검은색 Solid RGB로 교체
    #   3) 그 외 Base Color 연결 노드 → 회색 Solid RGB로 교체
    WHEEL_KEYWORDS = {'wheel', 'tire', 'tyre', '타이어', '휠', '실린더'}
    replaced = 0

    def _is_img_broken(img) -> bool:
        if img is None:
            return True
        abs_p = bpy.path.abspath(img.filepath) if img.filepath else ''
        return not (abs_p and os.path.isfile(abs_p)) and not img.has_data

    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # ── 1순위: MATERIAL_TEXTURE_MAP 수동 매핑 ──────────────────────────
        # 재질에 속한 모든 TEX_IMAGE 노드를 순회하여 깨진 것에 올바른 이미지 적용
        mapped_fname = MATERIAL_TEXTURE_MAP.get(mat.name)
        if mapped_fname:
            mapped_path = os.path.join(TEXTURE_DIR, mapped_fname)
            if os.path.isfile(mapped_path):
                new_img = bpy.data.images.load(mapped_path, check_existing=True)
                mat_fixed = False
                for node in nodes:
                    if node.type != 'TEX_IMAGE':
                        continue
                    node.image = new_img
                    if not mat_fixed:
                        replaced += 1
                        mat_fixed = True
            continue  # 매핑 대상 재질은 폴백 처리 불필요

        # ── 2·3순위: Base Color 연결 노드 Solid RGB 폴백 ──────────────────
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            continue
        base_color_input = bsdf.inputs.get('Base Color')
        if base_color_input is None:
            continue

        for lnk in list(base_color_input.links):
            src_node = lnk.from_node
            if src_node.type != 'TEX_IMAGE' or not _is_img_broken(src_node.image):
                continue

            mat_name_lower = mat.name.lower()
            is_wheel = any(kw in mat_name_lower for kw in WHEEL_KEYWORDS)
            if not is_wheel:
                for obj in bpy.data.objects:
                    if obj.type != 'MESH':
                        continue
                    if any(slot.material == mat for slot in obj.material_slots):
                        if any(kw in obj.name.lower() for kw in WHEEL_KEYWORDS):
                            is_wheel = True
                            break

            fallback_color = (0.02, 0.02, 0.02, 1.0) if is_wheel else (0.5, 0.5, 0.5, 1.0)
            rgb_node = nodes.new('ShaderNodeRGB')
            rgb_node.outputs[0].default_value = fallback_color
            rgb_node.location = (src_node.location.x, src_node.location.y)
            links.remove(lnk)
            links.new(rgb_node.outputs[0], base_color_input)
            nodes.remove(src_node)

            replaced += 1
            break

    # ── Phase 3: 투명도 재질 노드 수정 ───────────────────────────────────────
    # HASHED 투명도는 Cycles에서 극도로 노이지 → CLIP으로 교체
    # Alpha 소켓에 Color 출력이 연결된 경우 → 실제 PNG Alpha 출력으로 교체
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        if mat.blend_method not in ('HASHED', 'BLEND'):
            continue  # 투명도 미사용 재질은 건너뜀

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            continue

        alpha_input = bsdf.inputs.get('Alpha')
        if alpha_input is None:
            continue

        # Alpha 소켓에 연결된 링크 확인
        for lnk in list(alpha_input.links):
            src_node = lnk.from_node
            # Color 출력이 Alpha에 연결된 경우 → Alpha 출력으로 교체
            if src_node.type == 'TEX_IMAGE' and lnk.from_socket.name == 'Color':
                links.remove(lnk)
                links.new(src_node.outputs['Alpha'], alpha_input)

            # 동일 이미지를 위한 중복 TEX_IMAGE 노드 제거 (Color만 Alpha로 사용하던 노드)
            # Base Color와 연결된 노드와 같은 이미지를 쓰는 고아 노드 정리
            base_color_input = bsdf.inputs.get('Base Color')
            if base_color_input and base_color_input.links:
                main_img_node = base_color_input.links[0].from_node
                if (main_img_node.type == 'TEX_IMAGE' and
                        src_node != main_img_node and
                        src_node.type == 'TEX_IMAGE' and
                        src_node.image == main_img_node.image and
                        not any(lk for out in src_node.outputs for lk in out.links)):
                    nodes.remove(src_node)

        # HASHED → CLIP (Cycles에서 노이즈 없이 알파 컷아웃)
        if mat.blend_method == 'HASHED':
            mat.blend_method = 'CLIP'
            mat.alpha_threshold = 0.5


# =============================================================================
# PBR 재질 값 직접 설정 (부위별 차등 적용)
# =============================================================================

# ── 재질 이름 정확히 매칭하는 PBR 프로파일 ──────────────────────────────────
# 키: FBX에서 임포트된 정확한 재질명 (inspect_materials.py로 확인)
# 각 프로파일: (metallic, roughness, specular, clearcoat, clearcoat_rough)
PBR_PROFILES = {
    # 차체 도장 — 메인 바디 (반광 클리어코트)
    '포터 머터리얼':  {'metallic': 0.1, 'roughness': 0.4,  'specular': 0.5,
                     'clearcoat': 1.0, 'clearcoat_rough': 0.03},

    # 거친 금속 (하부 프레임, 적재함)
    '거친금속':       {'metallic': 0.9, 'roughness': 0.5,  'specular': 0.5,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 매끈한 크롬 금속 (범퍼, 손잡이)
    '금속매끈':       {'metallic': 1.0, 'roughness': 0.08, 'specular': 0.8,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 무광 검정 플라스틱 (트림, 범퍼 하부)
    '무광검정':       {'metallic': 0.0, 'roughness': 0.8,  'specular': 0.3,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 유광 검정 (그릴, 미러 하우징)
    '유광검정':       {'metallic': 0.1, 'roughness': 0.15, 'specular': 0.6,
                     'clearcoat': 0.8, 'clearcoat_rough': 0.05},

    # 형광주황 반사 스티커
    '형광주황':       {'metallic': 0.0, 'roughness': 0.4,  'specular': 0.5,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 후미등 (반투명 렌즈)
    ' 후미등이미지':  {'metallic': 0.1, 'roughness': 0.2,  'specular': 0.8,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 전조등 (투명 렌즈)
    '전조등사진':     {'metallic': 0.0, 'roughness': 0.05, 'specular': 0.9,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 타이어 (고무)
    '타이어이미지.':  {'metallic': 0.0, 'roughness': 0.9,  'specular': 0.2,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 번호판
    '번호판이미지':   {'metallic': 0.3, 'roughness': 0.3,  'specular': 0.5,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 바닥 그림자 (매테리얼)
    '매테리얼':       {'metallic': 0.0, 'roughness': 0.8,  'specular': 0.2,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},

    # 기타 기본 재질
    'Dots Stroke':    {'metallic': 0.0, 'roughness': 0.5,  'specular': 0.5,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},
    'Material':       {'metallic': 0.0, 'roughness': 0.5,  'specular': 0.5,
                     'clearcoat': 0.0, 'clearcoat_rough': 0.0},
}

# 위 프로파일에 매칭되지 않는 재질에 적용할 기본값
PBR_DEFAULT = {'metallic': 0.3, 'roughness': 0.1, 'specular': 0.7,
               'clearcoat': 0.0, 'clearcoat_rough': 0.0}


def apply_pbr_values():
    """
    FBX 임포트 후 기본값으로 남은 PBR 속성을 부위별로 차등 적용합니다.

    재질 이름에 따라 다른 프로파일을 적용하여 현실적인 질감을 구현합니다:
      - 차체 도장: Clearcoat으로 자동차 특유의 투명 코팅 광택
      - 크롬/금속: 높은 Metallic + 낮은 Roughness
      - 타이어: 비금속 + 매우 높은 Roughness (고무)
      - 무광 검정: 비금속 + 높은 Roughness (플라스틱 트림)

    해당 소켓에 이미 노드(텍스처맵 등)가 연결된 경우는 건드리지 않습니다.
    """
    updated = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            continue

        # ── 재질명 정확히 매칭 ──────────────────────────────────────────────
        profile = PBR_PROFILES.get(mat.name, PBR_DEFAULT)

        # ── PBR 값 적용 ──────────────────────────────────────────────────
        socket_map = [
            ('Metallic',            profile['metallic']),
            ('Roughness',           profile['roughness']),
            ('Specular IOR Level',  profile['specular']),     # Blender 4.x+
            ('Specular',            profile['specular']),     # Blender 3.x 호환
            ('Coat Weight',         profile['clearcoat']),    # Blender 4.x+
            ('Clearcoat',           profile['clearcoat']),    # Blender 3.x 호환
            ('Coat Roughness',      profile['clearcoat_rough']),  # Blender 4.x+
            ('Clearcoat Roughness', profile['clearcoat_rough']),  # Blender 3.x 호환
        ]

        changed = False
        for socket_name, value in socket_map:
            inp = bsdf.inputs.get(socket_name)
            if inp is None or inp.links:
                continue  # 소켓 없음 or 노드 연결됨 → 건드리지 않음
            inp.default_value = value
            changed = True

        if changed:
            updated += 1


# =============================================================================
# Domain Randomization — 매 프레임 호출
# =============================================================================

def randomize_domain():
    """
    매 프레임마다 환경·조명·재질을 무작위로 변경합니다 (Domain Randomization).

    변경 항목:
      1. HDR 환경맵  : HDRI_FILES 중 랜덤 선택, Z축 회전 랜덤
      2. 환경맵 강도 : 0.6 ~ 2.0 사이 랜덤
      3. Sun 방향    : 고도 20~80°, 방위각 0~360° 랜덤
      4. Sun 세기    : 0.5 ~ 6.0 사이 랜덤
      5. PBR 재질    : PBR_PROFILES 기준값에 ±노이즈 추가
      6. 바닥 재질   : 명도·거칠기 랜덤 (아스팔트·콘크리트·흙 범위)
    """
    def jitter(base, delta, lo=0.0, hi=1.0):
        return max(lo, min(hi, base + random.uniform(-delta, delta)))

    # ── 1. HDR 환경맵 교체 ────────────────────────────────────────────────
    if _DR_env_tex is not None and HDRI_FILES:
        hdr_path = random.choice(HDRI_FILES)
        # 같은 이미지면 재로드 생략, 다른 파일이면 교체
        if (_DR_env_tex.image is None or
                os.path.basename(_DR_env_tex.image.filepath) !=
                os.path.basename(hdr_path)):
            img = bpy.data.images.get(os.path.basename(hdr_path))
            if img is None:
                img = bpy.data.images.load(hdr_path)
            _DR_env_tex.image = img

        # HDR Z축 회전 (조명 방향 다양화)
        tree = bpy.context.scene.world.node_tree
        mapping = next((n for n in tree.nodes if n.type == 'MAPPING'), None)
        if mapping:
            mapping.inputs['Rotation'].default_value[2] = random.uniform(0, math.tau)

    # ── 2. 환경맵 강도 ────────────────────────────────────────────────────
    if _DR_bg_node is not None:
        _DR_bg_node.inputs['Strength'].default_value = random.uniform(0.6, 2.0)

    # ── 3 & 4. Sun 방향 · 세기 ───────────────────────────────────────────
    if _DR_sun_obj is not None:
        elev    = math.radians(random.uniform(20.0, 80.0))   # 고도각
        azimuth = math.radians(random.uniform(0.0, 360.0))   # 방위각
        _DR_sun_obj.rotation_euler = (
            math.pi / 2 - elev,   # X 회전 → 고도
            0.0,
            azimuth               # Z 회전 → 방위
        )
        _DR_sun_obj.data.energy = random.uniform(0.5, 6.0)

    # ── 5. PBR 재질 지터 ──────────────────────────────────────────────────
    # 재질별 지터 폭 (metallic_δ, roughness_δ)
    PBR_JITTER = {
        '포터 머터리얼': (0.05, 0.15),   # 차체: 도장 상태 다양
        '거친금속':      (0.05, 0.12),
        '금속매끈':      (0.03, 0.05),
        '무광검정':      (0.02, 0.10),
        '유광검정':      (0.05, 0.10),
        '형광주황':      (0.02, 0.08),
        '타이어이미지.': (0.02, 0.08),
    }
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        bsdf = next((n for n in mat.node_tree.nodes
                     if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            continue
        base = PBR_PROFILES.get(mat.name, PBR_DEFAULT)
        dm, dr = PBR_JITTER.get(mat.name, (0.02, 0.05))
        for socket_name, base_val, delta in [
            ('Metallic',  base['metallic'],  dm),
            ('Roughness', base['roughness'], dr),
        ]:
            inp = bsdf.inputs.get(socket_name)
            if inp and not inp.links:
                inp.default_value = jitter(base_val, delta)

    # ── 6. 바닥 재질 ──────────────────────────────────────────────────────
    if _DR_ground_mat is not None:
        bsdf = next((n for n in _DR_ground_mat.node_tree.nodes
                     if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf:
            # 아스팔트(0.04~0.12) · 콘크리트(0.12~0.25) · 흙(0.10~0.20) 범위
            brightness = random.uniform(0.04, 0.25)
            bsdf.inputs['Base Color'].default_value = (
                brightness, brightness, brightness * random.uniform(0.9, 1.1), 1.0)
            bsdf.inputs['Roughness'].default_value = random.uniform(0.75, 0.98)


# =============================================================================
# 카메라 포즈 샘플링 (Multi-View)
# =============================================================================

def sample_camera_pose(view_cat: str, truck_dims: dict,
                        truck_yaw_rad: float,
                        truck_empty: 'bpy.types.Object',
                        dist_max: float = None,
                        cam_z_min: float = None,
                        cam_z_max: float = None) -> tuple:
    """
    뷰 카테고리에 따라 카메라 위치와 Look-at 타겟을 샘플링합니다.

    좌표계 (truck_empty 로컬, 기하학적 중앙 = 원점):
        X : [-W/2, +W/2]   Y : [-L/2, +L/2]   Z : [-H/2, +H/2]
        뒷면 = Y: -L/2,  앞면 = Y: +L/2

    카메라는 트럭 로컬 공간에서 위치를 결정한 뒤 truck_yaw_rad 로 회전하여
    월드 좌표로 변환합니다. 이를 통해 트럭 회전에 상관없이 올바른 뷰를 확보합니다.

    Returns:
        cam_world  (mathutils.Vector): 카메라 월드 좌표
        look_world (mathutils.Vector): Look-at 타겟 월드 좌표 (노이즈 포함)
    """
    L = truck_dims['length']
    H = truck_dims['height']
    W = truck_dims['width']

    # 트럭 월드 중심 = truck_empty.location = (0, 0, H/2)
    truck_center_world = mathutils.Vector(truck_empty.location)

    # 카메라 높이는 월드 Z 기준 (지면=0), 로컬 Z 변환: cam_z_local = cam_z - H/2
    _z_min = cam_z_min if cam_z_min is not None else CAM_Z_MIN
    _z_max = cam_z_max if cam_z_max is not None else CAM_Z_MAX
    if _z_min > _z_max:
        _z_min, _z_max = _z_max, _z_min
    cam_z_world = random.uniform(_z_min, _z_max)
    cam_z_local = cam_z_world - H / 2.0
    _dist_max = dist_max if dist_max is not None else CAM_DIST_MAX

    if view_cat == 'rear':
        # 로컬 후면 중앙 (0, -L/2, 0) 기준, Y 음수 방향에서 접근
        dist = random.uniform(CAM_DIST_MIN, _dist_max)
        cam_x_local = random.uniform(-W * 1.5, W * 1.5)
        cam_y_local = -L / 2.0 - dist
        face_center_local = mathutils.Vector((0.0, -L / 2.0, 0.0))

    elif view_cat == 'front':
        # 로컬 전면 중앙 (0, +L/2, 0) 기준, Y 양수 방향에서 접근
        dist = random.uniform(CAM_DIST_MIN, _dist_max)
        cam_x_local = random.uniform(-W * 1.5, W * 1.5)
        cam_y_local = L / 2.0 + dist
        face_center_local = mathutils.Vector((0.0, L / 2.0, 0.0))

    elif view_cat == 'left':
        # 로컬 X 음수 방향에서 기하 중심을 바라봄
        dist = random.uniform(CAM_DIST_MIN, _dist_max)
        cam_x_local = -W / 2.0 - dist
        cam_y_local = random.uniform(-L / 2.0 - 1.0, L / 2.0 + 1.0)
        face_center_local = mathutils.Vector((0.0, 0.0, 0.0))

    else:  # 'right'
        # 로컬 X 양수 방향에서 기하 중심을 바라봄
        dist = random.uniform(CAM_DIST_MIN, _dist_max)
        cam_x_local = W / 2.0 + dist
        cam_y_local = random.uniform(-L / 2.0 - 1.0, L / 2.0 + 1.0)
        face_center_local = mathutils.Vector((0.0, 0.0, 0.0))

    # Look-at 노이즈 (거리 비례, 로컬 공간에서 적용)
    # v2: 수평 카메라 유지를 위해 Z 노이즈를 0으로 고정.
    #     Z 노이즈가 있으면 set_camera_look_at에서 제거되어 어차피 무효지만,
    #     look_world의 Z가 달라지면 레이블 저장 시 혼동이 생기므로 명시적으로 0으로 둠.
    n = dist * LOOKAT_NOISE_RATIO
    noise_local = mathutils.Vector((
        random.uniform(-n, n),
        random.uniform(-n, n),
        0.0,                      # Z 노이즈 제거 (수평 카메라 가정)
    ))
    # look_local.z = cam_z_local: 카메라와 동일 높이의 트럭 면을 조준.
    # set_camera_look_at에서 Z 차이를 제거하므로 결과는 동일하지만,
    # look_world를 레이블에 기록할 경우 의미상 올바른 값을 유지합니다.
    look_local = mathutils.Vector((
        face_center_local.x + noise_local.x,
        face_center_local.y + noise_local.y,
        cam_z_local,              # 카메라 높이와 동일 → look 방향이 수평
    ))

    # 트럭 로컬 → 월드 변환: world = truck_center_world + R_z(yaw) @ local
    rot_mat = mathutils.Matrix.Rotation(truck_yaw_rad, 3, 'Z')
    cam_local = mathutils.Vector((cam_x_local, cam_y_local, cam_z_local))
    cam_world  = truck_center_world + rot_mat @ cam_local
    look_world = truck_center_world + rot_mat @ look_local

    return cam_world, look_world


# =============================================================================
# 메인 생성 루프
# =============================================================================

def _render_depth_pass(scene, depth_dir: str, frame_idx: int, cam_obj):
    """
    Depth-only 렌더링 (Material Override 방식, Blender 5 호환).

    카메라 위치·방향을 상수로 셰이더에 주입하여 Z-depth(미터) 계산:
      depth = dot(surface_pos - cam_pos, cam_forward)
    배경 픽셀: film_transparent=True → R=0 → depth=0 (RealSense invalid)
    """
    # ── DepthCapture 머티리얼 ─────────────────────────────────────────────────
    mat = bpy.data.materials.new("DepthCapture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    geom = nodes.new('ShaderNodeNewGeometry')

    cam_pos_v = cam_obj.location
    cam_loc = nodes.new('ShaderNodeCombineXYZ')
    cam_loc.inputs['X'].default_value = float(cam_pos_v.x)
    cam_loc.inputs['Y'].default_value = float(cam_pos_v.y)
    cam_loc.inputs['Z'].default_value = float(cam_pos_v.z)

    cam_fwd_world = cam_obj.matrix_world.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0))
    cam_fwd_world.normalize()
    cam_fwd = nodes.new('ShaderNodeCombineXYZ')
    cam_fwd.inputs['X'].default_value = float(cam_fwd_world.x)
    cam_fwd.inputs['Y'].default_value = float(cam_fwd_world.y)
    cam_fwd.inputs['Z'].default_value = float(cam_fwd_world.z)

    sub = nodes.new('ShaderNodeVectorMath')
    sub.operation = 'SUBTRACT'
    links.new(geom.outputs['Position'], sub.inputs[0])
    links.new(cam_loc.outputs['Vector'], sub.inputs[1])

    dot = nodes.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    links.new(sub.outputs['Vector'], dot.inputs[0])
    links.new(cam_fwd.outputs['Vector'], dot.inputs[1])

    clamp = nodes.new('ShaderNodeMath')
    clamp.operation = 'MAXIMUM'
    clamp.inputs[1].default_value = 0.0
    links.new(dot.outputs['Value'], clamp.inputs[0])

    combine = nodes.new('ShaderNodeCombineXYZ')
    for c in ['X', 'Y', 'Z']:
        links.new(clamp.outputs['Value'], combine.inputs[c])

    emit = nodes.new('ShaderNodeEmission')
    links.new(combine.outputs['Vector'], emit.inputs['Color'])

    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(emit.outputs['Emission'], out.inputs['Surface'])

    # ── 렌더 설정 저장 → depth 렌더 → 복원 ──────────────────────────────────
    vl = scene.view_layers[0]
    vl.material_override = mat

    orig_samples     = scene.cycles.samples
    orig_denoise     = scene.cycles.use_denoising
    orig_filepath    = scene.render.filepath
    orig_format      = scene.render.image_settings.file_format
    orig_color_depth = scene.render.image_settings.color_depth
    orig_transparent = scene.render.film_transparent

    scene.cycles.samples                     = 1
    scene.cycles.use_denoising               = False
    scene.render.film_transparent            = True
    scene.render.image_settings.file_format  = 'OPEN_EXR'
    scene.render.image_settings.color_depth  = '32'
    scene.render.filepath = os.path.join(depth_dir, f'depth_{frame_idx:04d}.exr')

    bpy.ops.render.render(write_still=True)

    vl.material_override                     = None
    scene.cycles.samples                     = orig_samples
    scene.cycles.use_denoising               = orig_denoise
    scene.render.filepath                    = orig_filepath
    scene.render.image_settings.file_format  = orig_format
    scene.render.image_settings.color_depth  = orig_color_depth
    scene.render.film_transparent            = orig_transparent

    bpy.data.materials.remove(mat)


def _exr_to_npy(exr_path: str, npy_path: str, png_path: str,
                max_depth_m: float = 65.535):
    """
    Depth EXR (float, m) → RealSense D435i 동일 포맷으로 저장.
      .npy  : float32 (H,W), 미터, 0.0=invalid
      .png  : 16-bit grayscale, mm (시각화용)
    유효 범위: 0 ~ 65.535m (RealSense D435i 최대), 범위 밖/nan/inf → 0.0(invalid)
    """
    import array, numpy as np

    img = bpy.data.images.load(exr_path)
    W, H = img.size[0], img.size[1]
    raw = array.array('f', img.pixels[:])
    bpy.data.images.remove(img)

    depth = np.array(raw[0::4], dtype=np.float32).reshape(H, W)
    depth = np.flipud(depth)
    depth = np.where(np.isfinite(depth) & (depth <= max_depth_m), depth, 0.0).astype(np.float32)
    np.save(npy_path, depth)

    try:
        from PIL import Image as PILImage
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        PILImage.fromarray(depth_mm, mode='I;16').save(png_path)
    except ImportError:
        pass

    valid = depth[depth > 0]
    if len(valid):
        print(f"    depth {os.path.basename(npy_path)}: "
              f"{valid.min():.2f}–{valid.max():.2f}m  valid={len(valid)}/{depth.size}")


def _render_batch(env_cfg, n_frames, start_i, scene, cam_obj,
                  truck_empty, truck_dims, K):
    """
    하나의 환경(env_cfg)으로 씬을 유지한 채 n_frames 장을 렌더링합니다.
    반환: 다음 시작 인덱스
    """
    view_cats  = list(VIEW_WEIGHTS.keys())
    view_probs = list(VIEW_WEIGHTS.values())
    truck_center_world = mathutils.Vector(truck_empty.location)
    env_name = env_cfg['name']
    cam_z_min_world, cam_z_max_world = _resolve_camera_height_limits(
        truck_empty, truck_dims, env_cfg
    )
    frame_w = scene.render.resolution_x
    frame_h = scene.render.resolution_y
    i = start_i

    for _ in range(n_frames):
        # 이미 생성된 인덱스 건너뜀 (resume 시 중복 방지)
        while os.path.exists(os.path.join(IMAGE_DIR, f"image_{i:04d}.png")):
            i += 1
        attempt = 0
        while True:
            attempt += 1
            view_cat = random.choices(view_cats, weights=view_probs, k=1)[0]

            randomize_domain()

            truck_yaw_deg = random.uniform(0.0, 360.0)
            truck_yaw_rad = math.radians(truck_yaw_deg)
            truck_empty.rotation_euler = (0.0, 0.0, truck_yaw_rad)

            truck_center_world = mathutils.Vector(truck_empty.location)
            cam_pos, look_target = sample_camera_pose(
                view_cat, truck_dims, truck_yaw_rad, truck_empty,
                dist_max=env_cfg.get('cam_dist_max'),
                cam_z_min=cam_z_min_world,
                cam_z_max=cam_z_max_world)
            cam_obj.location = cam_pos
            set_camera_look_at(cam_obj, look_target)
            bpy.context.view_layer.update()

            truck_corners_world = get_truck_obb_corners(truck_dims, truck_empty)
            clearance = _evaluate_truck_view_clearance(
                scene, cam_obj, truck_empty, truck_dims, truck_corners_world
            )
            if not (
                clearance["center_visible"] and
                clearance["clear_ratio"] >= MIN_TRUCK_CLEAR_RATIO
            ):
                print(
                    f"  skip {i:04d} attempt {attempt:02d}"
                    f"  [{env_name:10s}|{view_cat:5s}]"
                    f"  occluded clear={clearance['visible_count']}/{clearance['sample_count']}"
                )
                if attempt >= MAX_CAMERA_SAMPLING_ATTEMPTS:
                    raise RuntimeError(
                        f"Could not find a clear camera view for frame {i:04d} "
                        f"in {env_name} after {attempt} attempts."
                    )
                continue

            cam_x, cam_y, cam_z = cam_pos.x, cam_pos.y, cam_pos.z
            dist = (cam_pos - truck_center_world).length
            truck_center_2d = world_to_image_pixel(scene, cam_obj, truck_center_world)
            if not (
                0.0 <= truck_center_2d[0] <= frame_w and
                0.0 <= truck_center_2d[1] <= frame_h
            ):
                print(
                    f"  skip {i:04d} attempt {attempt:02d}"
                    f"  [{env_name:10s}|{view_cat:5s}]"
                    f"  center_out=({truck_center_2d[0]:.1f},{truck_center_2d[1]:.1f})"
                )
                if attempt >= MAX_CAMERA_SAMPLING_ATTEMPTS:
                    raise RuntimeError(
                        f"Could not keep truck center in-frame for {i:04d} "
                        f"in {env_name} after {attempt} attempts."
                    )
                continue
            yaw_theta = compute_yaw_angle(cam_obj, truck_empty)
            corners_2d = [world_to_image_kp(scene, cam_obj, mathutils.Vector(c))
                          for c in truck_corners_world]

            rot_mat = mathutils.Matrix.Rotation(truck_yaw_rad, 3, 'Z')
            axes_2d = {
                "origin": truck_center_2d,
                "x_end": world_to_image_pixel(scene, cam_obj,
                             truck_center_world + rot_mat @ mathutils.Vector((1, 0, 0))),
                "y_end": world_to_image_pixel(scene, cam_obj,
                             truck_center_world + rot_mat @ mathutils.Vector((0, 1, 0))),
                "z_end": world_to_image_pixel(scene, cam_obj,
                             truck_center_world + mathutils.Vector((0, 0, 1))),
            }

            label_data = {
                "frame_id":      i,
                "view_category": view_cat,
                "truck_dims":    truck_dims,
                "metadata": {
                    "h_cam":           cam_z,
                    "cam_pos":         [cam_x, cam_y, cam_z],
                    "distance":        dist,
                    "truck_yaw_world": truck_yaw_deg,
                    "environment":     env_name,
                    "hdri":            (os.path.basename(_DR_env_tex.image.filepath)
                                        if _DR_env_tex and _DR_env_tex.image else "none"),
                    "sun_energy":      (_DR_sun_obj.data.energy if _DR_sun_obj else None),
                    "K_matrix":        K,
                    "depth_format": {
                        "file":          f"depth_{i:04d}.npy",
                        "dtype":         "float32",
                        "unit":          "meter",
                        "invalid_value": 0.0,
                        "max_range_m":   65.535,
                        "shape":         [scene.render.resolution_y,
                                          scene.render.resolution_x],
                    },
                },
                "ground_truth": {
                    "truck_center_2d": truck_center_2d,
                    "yaw_theta":       yaw_theta,
                    "3d_corners":      truck_corners_world,
                    "2d_corners":      corners_2d,
                    "axes_2d":         axes_2d,
                },
            }

            image_path = os.path.abspath(
                os.path.join(IMAGE_DIR, f"image_{i:04d}.png")
            )
            label_path = os.path.join(LABEL_DIR, f"label_{i:04d}.json")
            exr_path = os.path.join(DEPTH_DIR, f'depth_{i:04d}.exr')
            npy_path = os.path.join(DEPTH_DIR, f'depth_{i:04d}.npy')
            png_path = os.path.join(DEPTH_DIR, f'depth_{i:04d}.png')

            scene.render.filepath = image_path
            bpy.ops.render.render(write_still=True)

            # ── Depth EXR → .npy (float32 m, RealSense 동일) ─────────────────
            _render_depth_pass(scene, DEPTH_DIR, i, cam_obj)
            if os.path.exists(exr_path):
                _exr_to_npy(exr_path, npy_path, png_path)

            render_ok, render_stats = _rendered_image_is_usable(image_path)
            if not render_ok:
                for path in (image_path, label_path, exr_path, npy_path, png_path):
                    if os.path.exists(path):
                        os.remove(path)
                print(
                    f"  skip {i:04d} attempt {attempt:02d}"
                    f"  [{env_name:10s}|{view_cat:5s}]"
                    f"  dark mean={render_stats['mean_luma']:.2f}"
                    f" max={render_stats['max_channel']}"
                )
                if attempt >= MAX_CAMERA_SAMPLING_ATTEMPTS:
                    raise RuntimeError(
                        f"Could not render a usable frame for {i:04d} "
                        f"in {env_name} after {attempt} attempts."
                    )
                continue

            with open(label_path, 'w', encoding='utf-8') as fp:
                json.dump(label_data, fp, indent=2, ensure_ascii=False)

            attempt_suffix = f"  tries={attempt}" if attempt > 1 else ""
            print(
                f"  {i:04d}  [{env_name:10s}|{view_cat:5s}]"
                f"  cam=({cam_x:+5.2f},{cam_y:+6.2f},{cam_z:.2f})m"
                f"  dist={dist:5.2f}m  yaw={yaw_theta:6.1f}°"
                f"{attempt_suffix}"
            )
            i += 1
            break

    return i


def main():
    global HDRI_FILES
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    HDRI_FILES = _collect_hdri_files()
    print(f"  HDRI: {len(HDRI_FILES)} files")

    # ── 이어서 생성: 기존 레이블에서 환경별 카운트 파악 ────────────────────
    existing_by_env = {e['name']: 0 for e in ENV_CONFIGS}
    existing_indices = []
    for fn in sorted(os.listdir(LABEL_DIR)):
        if not (fn.startswith('label_') and fn.endswith('.json')):
            continue
        try:
            with open(os.path.join(LABEL_DIR, fn), encoding='utf-8') as f:
                d = json.load(f)
            env = d.get('metadata', {}).get('environment', '')
            if env in existing_by_env:
                existing_by_env[env] += 1
            existing_indices.append(d['frame_id'])
        except Exception:
            pass

    already = sum(existing_by_env.values())
    start_i = _first_missing_index(existing_indices)

    if already > 0:
        print(f"Resume: {already}/{NUM_IMAGES} 이미 생성됨 "
              f"(next index: {start_i})")
        print("  환경별 기존: " +
              ", ".join(f"{k}={v}" for k, v in existing_by_env.items()))

    if already >= NUM_IMAGES:
        print("이미 완료됨.")
        return

    # 환경별 목표 수 계산 후 기존 생성분 차감
    total   = NUM_IMAGES
    weights = [e['weight'] for e in ENV_CONFIGS]
    w_sum   = sum(weights)
    targets = [round(total * w / w_sum) for w in weights]
    targets[-1] = total - sum(targets[:-1])
    counts  = [max(0, t - existing_by_env.get(e['name'], 0))
               for e, t in zip(ENV_CONFIGS, targets)]

    print(f"Generating {sum(counts)} more images: " +
          ", ".join(f"{e['name']}×{c}" for e, c in zip(ENV_CONFIGS, counts)))

    i = start_i
    total_generated = 0

    try:
        for env_cfg, n in zip(ENV_CONFIGS, counts):
            if n <= 0:
                continue
            env_name = env_cfg['name']
            map_name = env_cfg.get('map_name')
            print(f"\n── {env_name}  {n}장 ──")

            # 씬 초기화 및 트럭/환경 셋업
            clear_scene()
            setup_render_settings()
            truck_empty, truck_dims = import_truck()
            fix_missing_textures()
            apply_pbr_values()

            if map_name:
                if env_cfg.get('map_offset'):
                    MAP_CONFIGS[map_name]['center_offset'] = env_cfg['map_offset']
                import_environment_map(map_name)
                cfg = MAP_CONFIGS.get(map_name, {})
                if cfg.get('interior'):
                    setup_interior_lighting()
                else:
                    setup_lighting()
            else:
                create_ground_plane()
                setup_lighting()

            cam_obj = create_camera()
            scene   = bpy.context.scene
            apply_env_truck_pose(scene, truck_empty, truck_dims, env_cfg)
            K = get_camera_intrinsic_matrix(scene, cam_obj)
            bpy.context.view_layer.update()

            i = _render_batch(env_cfg, n, i, scene, cam_obj,
                              truck_empty, truck_dims, K)
            total_generated += n

    except KeyboardInterrupt:
        print(f"\nStopped. {total_generated} images generated (last index: {i-1}).")

    print(f"\nDone. Total generated: {total_generated}, last index: {i-1}")


# =============================================================================
# 정후방 테스트 렌더링
# =============================================================================

def _run_test_views(views=('rear', 'front', 'left', 'right'), map_name=None):
    """
    지정한 뷰(들)을 정확한 위치에서 렌더링하고 레이블도 저장합니다.
    트럭 yaw=0 고정 (테스트 기준 자세).
    출력: datasets/v1/images/test_<view>.png
          datasets/v1/labels/test_<view>.json

    map_name: None(HDRI) | 'warehouse' | ... 3D 환경 맵 이름
    """
    global HDRI_FILES
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)
    clear_scene()
    setup_render_settings()
    truck_empty, truck_dims = import_truck()
    fix_missing_textures()
    apply_pbr_values()

    if map_name:
        # 3D 환경 맵: 자체 바닥이 있으므로 shadow catcher 생략
        import_environment_map(map_name)
        cfg = MAP_CONFIGS.get(map_name, {})
        if cfg.get('interior'):
            setup_interior_lighting()
        else:
            # 실외 맵: HDRI + Sun 초기화
            if not HDRI_FILES:
                HDRI_FILES = _collect_hdri_files()
            setup_lighting()
    else:
        # HDRI 모드: shadow catcher 바닥 + HDR 환경
        if not HDRI_FILES:
            HDRI_FILES = _collect_hdri_files()
        create_ground_plane()
        setup_lighting()

    cam_obj = create_camera()
    test_env_cfg = {}
    if map_name == 'takamatsu':
        test_env_cfg = TAKAMATSU_ENV_CONFIG
    apply_env_truck_pose(bpy.context.scene, truck_empty, truck_dims, test_env_cfg)

    scene = bpy.context.scene
    K = get_camera_intrinsic_matrix(scene, cam_obj)

    L = truck_dims['length']
    H = truck_dims['height']
    W = truck_dims['width']

    # 트럭 yaw=0 고정
    truck_empty.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    truck_center_world = mathutils.Vector(truck_empty.location)  # (0, 0, H/2)

    # 뷰별 카메라 위치 & Look-at (트럭 yaw=0, 5m 거리)
    TEST_DIST = 5.0
    VIEW_POSES = {
        'rear':  (mathutils.Vector((0.0, -L/2 - TEST_DIST, H/2)),
                  mathutils.Vector((0.0, -L/2,              H/2))),
        'front': (mathutils.Vector((0.0,  L/2 + TEST_DIST, H/2)),
                  mathutils.Vector((0.0,  L/2,              H/2))),
        'left':  (mathutils.Vector((-W/2 - TEST_DIST, 0.0, H/2)),
                  mathutils.Vector((0.0,               0.0, H/2))),
        'right': (mathutils.Vector(( W/2 + TEST_DIST, 0.0, H/2)),
                  mathutils.Vector((0.0,               0.0, H/2))),
    }

    for view_cat in views:
        if view_cat not in VIEW_POSES:
            print(f"  Unknown view: {view_cat}, skipping.")
            continue

        randomize_domain()

        cam_pos, look_at = VIEW_POSES[view_cat]
        cam_obj.location = cam_pos
        set_camera_look_at(cam_obj, look_at)
        bpy.context.view_layer.update()

        truck_corners_world = get_truck_obb_corners(truck_dims, truck_empty)
        cam_x, cam_y, cam_z = cam_pos.x, cam_pos.y, cam_pos.z
        dist = (cam_pos - truck_center_world).length
        truck_center_2d = world_to_image_pixel(scene, cam_obj, truck_center_world)
        yaw_theta  = compute_yaw_angle(cam_obj, truck_empty)
        corners_2d = [world_to_image_kp(scene, cam_obj, mathutils.Vector(c))
                      for c in truck_corners_world]
        axes_2d = {
            "origin": truck_center_2d,
            "x_end": world_to_image_pixel(scene, cam_obj,
                         truck_center_world + mathutils.Vector((1.0, 0.0, 0.0))),
            "y_end": world_to_image_pixel(scene, cam_obj,
                         truck_center_world + mathutils.Vector((0.0, 1.0, 0.0))),
            "z_end": world_to_image_pixel(scene, cam_obj,
                         truck_center_world + mathutils.Vector((0.0, 0.0, 1.0))),
        }

        label_data = {
            "frame_id":      f"test_{view_cat}",
            "view_category": view_cat,
            "truck_dims":    truck_dims,
            "metadata": {
                "h_cam":           cam_z,
                "cam_pos":         [cam_x, cam_y, cam_z],
                "distance":        dist,
                "truck_yaw_world": 0.0,
                "environment":     (map_name if map_name else
                                    (os.path.basename(_DR_env_tex.image.filepath)
                                     if _DR_env_tex and _DR_env_tex.image else "procedural")),
                "sun_energy":      (_DR_sun_obj.data.energy if _DR_sun_obj else None),
                "K_matrix":        K,
            },
            "ground_truth": {
                "truck_center_2d": truck_center_2d,
                "yaw_theta":       yaw_theta,
                "3d_corners":      truck_corners_world,
                "2d_corners":      corners_2d,
                "axes_2d":         axes_2d,
            },
        }

        img_path = os.path.join(IMAGE_DIR, f"test_{view_cat}.png")
        lbl_path = os.path.join(LABEL_DIR, f"test_{view_cat}.json")

        scene.render.filepath = os.path.abspath(img_path)
        bpy.ops.render.render(write_still=True)
        with open(lbl_path, 'w', encoding='utf-8') as fp:
            json.dump(label_data, fp, indent=2, ensure_ascii=False)
        print(f"  [{view_cat}] image -> {img_path}")
        print(f"  [{view_cat}] label -> {lbl_path}")


# =============================================================================
# 엔트리 포인트
# =============================================================================

if __name__ == "__main__":
    # blender -- --num-images N   : 이미지 수 오버라이드
    # blender -- --test-rear      : 정확한 정후방 1장만 렌더링
    # blender -- --test-front     : 정확한 정전방 1장만 렌더링 (PBR 확인용)
    _argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []

    if '--num-images' in _argv:
        _idx = _argv.index('--num-images')
        try:
            NUM_IMAGES = int(_argv[_idx + 1])
        except (IndexError, ValueError):
            pass

    if '--dataset-version' in _argv:
        _idx = _argv.index('--dataset-version')
        try:
            configure_output_paths(_argv[_idx + 1])
        except IndexError:
            pass

    # --map <name> 파라미터 파싱
    _map_name = None
    if '--map' in _argv:
        _map_name = _argv[_argv.index('--map') + 1]

    # --city-only : city 환경만 생성 (city_A / city_B 균등 분배)
    if '--city-only' in _argv:
        city_envs = [e for e in ENV_CONFIGS if e.get('map_name') == 'city']
        w_each = 1.0 / len(city_envs)
        for e in city_envs:
            e['weight'] = w_each
        ENV_CONFIGS[:] = city_envs
    elif '--takamatsu-only' in _argv:
        ENV_CONFIGS[:] = [dict(TAKAMATSU_ENV_CONFIG)]
    elif '--cnr-middle-only' in _argv:
        ENV_CONFIGS[:] = [dict(CNR_MIDDLE_NOWHERE_ENV_CONFIG)]

    if '--test-warehouse' in _argv:
        _run_test_views(('rear', 'front', 'left', 'right'), map_name='warehouse')
    elif '--test-city' in _argv:
        _run_test_views(('rear', 'front', 'left', 'right'), map_name='city')
    elif '--test-all' in _argv:
        _run_test_views(('rear', 'front', 'left', 'right'), map_name=_map_name)
    elif '--test-rear' in _argv:
        _run_test_views(('rear',), map_name=_map_name)
    elif '--test-front' in _argv:
        _run_test_views(('front',), map_name=_map_name)
    elif '--test-left' in _argv:
        _run_test_views(('left',), map_name=_map_name)
    elif '--test-right' in _argv:
        _run_test_views(('right',), map_name=_map_name)
    else:
        main()
