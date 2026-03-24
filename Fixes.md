# Fixes

이 파일은 코드 변경이 발생할 때마다 커밋 단위로 누적 기록한다.
각 항목에는 변경 이유, 실제 수정 내용, 검증 또는 학습 로그 위치를 함께 남긴다.

## 2026-03-23 - fix: handle non-contiguous tensors in SMOKE box encoding and add geometry training logs

### Reason
- `logs/geometry_smoke_seed42.log` 기준 2026-03-22 13:08 학습이 `SMOKE-master/smoke/modeling/smoke_coder.py`의 `encode_box3d()`에서 중단됐다.
- 원인은 `view()`가 non-contiguous tensor에 적용되면서 `RuntimeError: view size is not compatible with input tensor's size and stride`가 발생한 것이다.

### Code Change
- `SMOKE-master/smoke/modeling/smoke_coder.py`
- `dims`, `locs`, `box_3d_object`를 다루는 부분의 `view()` 호출을 `reshape()`로 바꿨다.
- 이 변경으로 contiguous 여부에 덜 민감하게 3D box encoding 경로를 유지할 수 있다.

### Training Logs Included
- `logs/geometry_smoke_seed42.log`
  - 최초 실행 실패 로그
- `logs/geometry_smoke_seed42_retry.log`
  - 수정 후 재실행 로그
- `results/geometry/seed_42/log.txt`
  - 학습기에서 기록한 상세 로그
- `results/geometry/seed_42/run_meta.json`
  - 실행 파라미터 메타 정보

### Training Summary
- 재실행 로그는 `results/geometry/seed_42` 기준 seed `42`, batch `8`, max iteration `25000` 설정으로 시작됐다.
- 마지막 기록 시각은 2026-03-22 16:29:06이며, `iter 22580`까지 진행됐다.
- 마지막 로그 값은 `loss 1.0192 (avg 1.4875)`, `hm_loss 0.0833 (avg 0.4329)`, `reg_loss 0.7655 (avg 1.0546)`이다.
- 현재 관련 학습 프로세스는 실행 중이지 않다.

### Artifact Policy
- `model_0010000.pth`, `model_0018000.pth` 체크포인트는 각각 약 224MB라 일반 GitHub push 대상에서는 제외했다.
- 로그와 실행 메타 정보만 저장소에 포함했다.

### Future Rule
- 이후 코드 변경이 생기면 이 파일에 새 섹션을 추가한다.
- 섹션 제목은 커밋 단위로 작성하고, 최소한 `Reason`, `Code Change`, `Verification or Logs`를 남긴다.

## 2026-03-23 - feat: add local AP@0.70 evaluation workflow for SMOKE checkpoints and record geometry checkpoint results

### Reason
- 요청 사항은 baseline, geometry 모델의 저장된 체크포인트를 기준으로 IoU 0.7 AP를 iter 단위로 비교하는 것이었다.
- 로컬 점검 결과 geometry 체크포인트는 `results/geometry/seed_42/`에 있었지만, baseline 체크포인트 `model_*.pth`는 이 머신에서 찾을 수 없었다.
- 공식 SMOKE의 eval-only 경로는 현재 데이터셋 구조와 바로 맞지 않았다.
- 기본 `paths_catalog.py`는 `kitti_test -> datasets/kitti/testing/`을 가리켜 `val` split 평가 시 `testing/ImageSets/val.txt`를 찾다가 실패했다.
- 또한 저장소에는 KITTI native evaluator 소스가 없어 AP 계산 바이너리 `evaluate_object_3d_offline`를 직접 빌드할 수 없는 상태였다.
- 공식 `kitti_eval.py`는 마지막 단계에서 `../smoke/data/datasets/evaluation/kitti/kitti_eval`로 `chdir`하는데, 현재 실행 기준에서는 이 상대 경로가 맞지 않아 추론은 끝나도 평가 단계에서 종료됐다.

### Code Change
- `train/paths_catalog_val.py`
- validation AP 평가 전용 dataset catalog를 추가했다.
- `kitti_test`를 `datasets/kitti/training/`으로 매핑해서 공식 SMOKE eval-only가 `training/ImageSets/val.txt`와 `training/label_2`를 사용하도록 맞췄다.

- `SMOKE-master/smoke/data/datasets/evaluation/kitti/kitti_eval/evaluate_object_3d_offline.cpp`
- `SMOKE-master/smoke/data/datasets/evaluation/kitti/kitti_eval/mail.h`
- KITTI native evaluator 소스와 헤더를 추가해 로컬에서 AP를 직접 계산할 수 있게 했다.
- 컴파일은 conda include 경로(`/home/dy-jang/anaconda3/include`)를 사용해 boost 헤더를 찾도록 처리했다.
- 참고로 이 경로는 `.gitignore`의 `datasets/` 패턴에 걸리므로 커밋 시 force add가 필요하다.

### Evaluation Workflow
- geometry 체크포인트 `model_0010000.pth`, `model_0018000.pth`에 대해 공식 SMOKE `--eval-only` 추론을 먼저 실행했다.
- 추론 결과 prediction txt를 만든 뒤, native evaluator를 직접 호출해 AP를 계산했다.
- plot 생성 단계에서 `gnuplot`, `pdfcrop`가 없어 경고가 있었지만, AP 수치 텍스트 출력 자체는 정상 생성됐다.

### Results Included
- `results/ap_eval/ap70_summary.json`
  - iter 단위 AP 요약 JSON
- `results/ap_eval/geometry_ap70_by_iter.png`
  - iter 단위 시각화 PNG
- `results/ap_eval/geometry/model_0010000_ap.txt`
  - geometry 10000 iter AP 원문
- `results/ap_eval/geometry/model_0018000_ap.txt`
  - geometry 18000 iter AP 원문
- `results/ap_eval/geometry/model_0010000_eval.log`
  - geometry 10000 iter eval-only 실행 로그
- `results/ap_eval/geometry/model_0018000_eval.log`
  - geometry 18000 iter eval-only 실행 로그
- `results/ap_eval/geometry/model_0010000/log.txt`
  - 공식 SMOKE output dir 내부 로그
- `results/ap_eval/geometry/model_0018000/log.txt`
  - 공식 SMOKE output dir 내부 로그

### Measured AP@0.70
- geometry `iter 10000`
- 2D AP: `easy 0.0000 / moderate 0.0000 / hard 0.4091`
- BEV AP: `easy 0.0000 / moderate 0.0000 / hard 0.0000`
- 3D AP: `easy 0.0000 / moderate 0.0000 / hard 0.0000`

- geometry `iter 18000`
- 2D AP: `easy 0.0000 / moderate 0.1404 / hard 1.5942`
- BEV AP: `easy 0.0000 / moderate 0.0000 / hard 0.0000`
- 3D AP: `easy 0.0000 / moderate 0.0000 / hard 0.0000`

### Baseline Status
- baseline 관련 로그는 있었지만 이 머신에서 실제 baseline 체크포인트 `model_*.pth`는 찾지 못했다.
- 따라서 이번 결과물은 geometry만 실측했고, `results/ap_eval/ap70_summary.json`에도 baseline 상태를 `missing_checkpoints`로 남겼다.
- baseline 체크포인트 경로가 확보되면 같은 평가 흐름으로 동일 그래프에 바로 추가할 수 있다.

### Artifact Policy
- 중간 prediction txt와 임시 eval 폴더는 최종 커밋에서 제외했다.
- 최종 비교에 필요한 요약 결과, 원문 AP 텍스트, 실행 로그, 시각화만 남겼다.
- native evaluator 바이너리 `evaluate_object_3d_offline` 자체는 생성 산출물이라 커밋 대상에서 제외했다.

### Verification or Logs
- geometry checkpoint 추론 후 native evaluator 직접 실행으로 AP 텍스트를 생성했다.
- 최종 요약은 `results/ap_eval/ap70_summary.json`에 저장했다.
- 최종 시각화는 `results/ap_eval/geometry_ap70_by_iter.png`에 저장했다.

## 2026-03-24 - chore: sync local geometry/baseline research artifacts, validators, and qualitative results to GitHub

### Reason
- 로컬 작업트리에는 GitHub `origin/main` 이후 반영된 geometry known-dimension 경로, KITTI 변환 검증기, 강화된 시각화 코드, 학습 로그, supervisor 스크립트, 정성/정량 비교 결과가 누적되어 있었다.
- 요청 사항은 가중치 파일과 데이터셋처럼 무거운 산출물을 제외하고 현재 로컬 상태를 GitHub에 동기화하는 것이었다.

### Code Change
- `SMOKE-master/smoke/data/datasets/kitti.py`
- `SMOKE-master/smoke/modeling/heads/smoke_head/loss.py`
- `SMOKE-master/smoke/modeling/heads/smoke_head/inference.py`
- geometry 경로가 sample-wise known dimensions를 train/val/inference에 사용할 수 있도록 로컬 반영 상태를 동기화했다.

- `train/validate_kitti_conversion.py`
- `train/visualize_kitti_mapping_and_predictions.py`
- `tools/inspect_smoke_predictions.py`
- KITTI 변환 검증, GT/prediction overlay, qualitative review를 위한 로컬 도구를 포함했다.

- `logs/supervise_smoke_runs.sh`
- `logs/supervise_smoke_runs.log`
- `logs/geometry_seed42.log`
- `logs/baseline_seed42.log`
- 중단 복구용 supervisor와 최신 학습 로그 snapshot을 포함했다.

### Results and Logs Included
- `results/kitti_conversion_validation/*.json`
- 최신 KITTI 변환 검증 요약 결과를 포함했다.
- `results/kitti_vis_compare/**`
- geometry/baseline 샘플 추론과 GT 비교 시각화 PNG를 포함했다.
- `results/single_infer/**`
- 단일/소수 샘플 추론 결과 txt와 eval 로그를 포함했다.
- `results/iter6000_compare/iter6000_three_metrics_summary.json`
- baseline/geometry `iter 6000`의 3개 샘플 비교 지표 요약을 포함했다.
- `results/baseline/seed_42/log.txt`
- `results/baseline/seed_42/run_meta.json`
- `results/geometry/seed_42/log.txt`
- `results/geometry/seed_42/last_checkpoint`
- 현재 학습 진행 상태 snapshot을 포함했다.

### Local Cleanup Reflected
- 이전 geometry AP 실험 산출물 일부를 로컬에서 제거한 현재 상태를 반영했다.
- 삭제 반영 항목:
  - `logs/geometry_run.log`
  - `logs/geometry_smoke_seed42.log`
  - `logs/geometry_smoke_seed42_retry.log`
  - `results/ap_eval/geometry/model_0010000/*`
  - `results/ap_eval/geometry/model_0018000/*`
  - `results/ap_eval/geometry_ap70_by_iter.png`
  - `results/ablation_study/seed_42/geometry/history.json`

### Artifact Policy
- `.gitignore` 기준으로 `datasets/`, `*.pth`, `*.pt`, `*.ckpt`는 계속 제외했다.
- 따라서 학습 체크포인트, 모델 가중치, 원본/변환 데이터셋은 이번 동기화 대상에 포함하지 않았다.

### Verification or Logs
- `git status --short --untracked-files=all`로 동기화 대상 파일을 점검했다.
- `git fetch origin` 후 `origin/main` 대비 로컬 작업트리를 비교해 최신 원격 변경과 로컬 산출물을 함께 정리했다.

## 2026-03-24 - docs: split repository landing page and detailed structure guide

### Reason
- 요청 사항은 프로젝트 전체 구조를 설명하는 상세 문서와, GitHub 프로젝트 첫 화면에 표시할 간결한 문서를 분리하는 것이었다.
- 기존 `README.md`는 실행 방법 중심이어서 저장소 전체 구조와 역할 분리가 한눈에 보이기 어려웠다.

### Code Change
- `README.md`
- GitHub 첫 화면용 요약 문서로 재구성했다.
- 프로젝트 개요, 빠른 시작 경로, 핵심 진입점, 상세 문서 링크를 중심으로 정리했다.

- `Readme.md`
- 저장소 구조 설명용 상세 문서를 새로 추가했다.
- 루트 파일, `SMOKE-master/`, `train/`, `tools/`, `results/`, `logs/`, `external/` 등 실제 디렉터리 역할과 현재 권장 workflow를 문서화했다.

### Verification or Logs
- 현재 저장소의 최상위 파일/디렉터리와 `train/`, `tools/` 구성 기준으로 문서를 작성했다.
- 상세 문서에는 현재 권장 엔트리포인트와 산출물 위치를 함께 정리했다.
