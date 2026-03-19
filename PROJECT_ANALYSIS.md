# 프로젝트 분석 정리

## 1. 이 프로젝트는 무엇인가

이 프로젝트는 **Blender로 생성한 합성 트럭 데이터셋을 이용해 단안(monocular) 3D 바운딩 박스 추정을 연구하는 실험 프로젝트**입니다.  
핵심 주제는 다음 질문으로 보입니다.

> 카메라 높이와 투영 기하를 알고 있을 때, 깊이를 직접 학습하지 않고도 트럭의 3D 위치를 더 단순하고 안정적으로 복원할 수 있는가?

즉, 일반적인 SMOKE/CenterNet 계열의 단안 3D 검출 방식과,  
기하 제약식 `Z = fy * h_cam / (v_c - cy)`를 활용하는 방식을 비교하는 **ablation study(절제 연구)**가 프로젝트의 중심입니다.

## 2. 확인한 근거

- [README.md](/Users/doyoung/Documents/Blender/README.md): 프로젝트 개요, 데이터셋 설명, 4개 모델 변형, 학습 설정, 실행 방법이 정리되어 있음
- [generate_synthetic_dataset.py](/Users/doyoung/Documents/Blender/generate_synthetic_dataset.py): Blender 기반 합성 데이터셋 생성 스크립트
- [train/models.py](/Users/doyoung/Documents/Blender/train/models.py): SMOKE 스타일 4종 모델(`baseline`, `geometry`, `baseline_depth`, `geometry_aux`) 정의
- [train/dataset.py](/Users/doyoung/Documents/Blender/train/dataset.py): `datasets/v3` 기준 데이터 로딩, RGB/Depth/3D box/카메라 파라미터 처리
- [train/ablation_study.py](/Users/doyoung/Documents/Blender/train/ablation_study.py): 다중 시드 절제 연구 실행기
- [results/ablation_study/summary.json](/Users/doyoung/Documents/Blender/results/ablation_study/summary.json): 실제 실험 결과 요약 일부 존재

## 3. 현재 폴더 구조를 바탕으로 본 프로젝트 성격

이 저장소는 크게 4개 축으로 구성되어 있습니다.

### A. 합성 데이터 생성

- `generate_synthetic_dataset.py`
- `regen_missing.py`
- `cleanup_dataset.py`
- `update_split.py`
- `analyze_dataset.py`
- `visualize_depth.py`
- `visualize_labels.py`

역할:

- Blender에서 현대 포터 트럭 FBX 모델을 불러옴
- 도시/창고 맵, HDRI, 조명, 카메라 위치, 거리, 시점을 랜덤화함
- RGB 이미지, depth, 3D 박스 코너, 카메라 내부 파라미터, yaw, view category 등을 라벨로 저장함
- 데이터셋 품질 점검과 시각화 도구도 함께 포함됨

### B. 학습/실험 코드

- `train/models.py`
- `train/smoke_loss.py`
- `train/smoke_trainer.py`
- `train/metrics.py`
- `train/ablation_study.py`
- `train/dataset.py`

역할:

- ResNet-34 기반 SMOKE 스타일 검출 모델 구성
- heatmap, offset, yaw, depth, 3D corner 관련 손실 계산
- baseline 대비 geometry 제약 방식의 성능 차이 비교
- 단일 실행과 다중 시드 절제 연구를 모두 지원

### C. 데이터/에셋

- `datasets/`
- `hyundai-porter-truck/`
- `map/`
- `hdri/`
- `yolo26n-pose.pt`

역할:

- 여러 버전의 데이터셋 실험 흔적이 남아 있음 (`v0_beta`, `v1`, `v2`, `v3`, `v3_test`, `v4`)
- 실제 학습 타깃은 코드상 `datasets/v3`인 것으로 보임
- 트럭 FBX/텍스처와 배경 맵, HDRI가 합성 데이터 생성에 사용됨
- `yolo26n-pose.pt`는 초기 실험용 백본 또는 관련 실험 자산으로 보임

### D. 결과/문서

- `results/`
- `logs/`
- `docs/`
- `paper/`

역할:

- 절제 연구 결과 그래프, 체크포인트, history JSON 보관
- 샘플 이미지와 라벨 시각화 예시 제공
- 참고 논문 PDF(`SMOKE`, `MonoGround`, `GUPNet`)가 포함되어 있어, 구현이 논문 재현/변형 성격을 가짐

## 4. 실제로 파악한 핵심 파이프라인

프로젝트 흐름은 아래와 같습니다.

1. Blender에서 합성 장면을 렌더링해 트럭 데이터셋 생성
2. RGB, depth, 2D/3D box, 카메라 파라미터, yaw, 시점 정보 저장
3. `train/dataset.py`가 이를 640x640 letterbox 기준으로 학습 데이터로 변환
4. `train/models.py`의 4개 모델 변형을 학습
5. `train/ablation_study.py`로 seed별 반복 실험 수행
6. `results/`와 `logs/`에 결과 저장 및 시각화

즉, 이 프로젝트는 단순한 모델 학습 코드가 아니라:

- **합성 데이터 생성**
- **3D detection 학습**
- **기하 기반 방법 검증**
- **논문식 절제 연구 정리**

까지 한 저장소 안에서 수행하는 **연구형 컴퓨터 비전 프로젝트**입니다.

## 5. 현재 상태에서 보이는 구체적 특징

- README 기준 프로젝트명은 `TruckDetection`
- 대상 객체는 현대 포터 계열 트럭
- 문제 설정은 **monocular 3D bounding box estimation**
- 데이터셋은 합성(synthetic) 중심이며 실제 RealSense 관련 폴더도 일부 존재
- 실험 비교축은 다음 4개 모델임
  - `baseline`
  - `geometry`
  - `baseline_depth`
  - `geometry_aux`
- 핵심 가설은 깊이 추정을 전부 네트워크가 배우게 하는 대신, 카메라 높이와 바닥 접점 정보를 이용해 깊이를 기하식으로 복원하는 것이 도움이 되는지 검증하는 것

## 6. 저장소에서 확인한 현재 진행 상황

- `results/ablation_study/`와 `results/smoke_ablation/`가 존재해 학습/비교 실험이 실제로 수행된 흔적이 있음
- `results/ablation_study/summary.json`에는 현재 `geometry_aux` 결과만 기록되어 있어, 전체 4종 5시드 실험이 아직 완전히 끝난 상태는 아닐 수 있음
- Git 상태상 아래 파일은 이미 수정 중이므로 활발히 실험/개선 중인 작업 저장소로 보임
  - `train/ablation_study.py`
  - `train/models.py`
  - `train/smoke_loss.py`

## 7. 한 줄 요약

이 프로젝트는 **Blender로 만든 합성 트럭 데이터셋을 기반으로, SMOKE 계열 단안 3D 객체 검출 모델에 기하 제약을 도입했을 때의 효과를 비교 분석하는 연구용 실험 저장소**입니다.

## 8. 참고 메모

- 실제 학습 기준 데이터셋 루트는 현재 코드상 `datasets/v3`로 보입니다.
- README와 일부 코드에는 실험 설정 차이의 흔적이 있어, 초기 버전과 최신 구현이 혼재했을 가능성이 있습니다.
- 따라서 이 저장소는 “완성된 제품”보다는 “실험과 개선이 이어지는 연구/개발 작업공간”으로 보는 것이 가장 정확합니다.
