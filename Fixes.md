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
