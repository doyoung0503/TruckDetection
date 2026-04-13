# Current Training Status

이 starter 폴더는 `5-sample raw + 5-sample converted + baseline/geometry launch code`를 묶은 재현용 패키지입니다.

현재 full clean re-export 데이터셋 기준으로 확인된 상태는 다음과 같습니다.

- clean re-export: 통과
- FCOS3D baseline full run: 학습은 끝까지 진행됐지만 후반 `NaN`으로 붕괴
- FCOS3D geometry full run: 이 starter를 정리하는 쪽으로 우선순위를 바꾸면서 아직 다시 돌리지 않음

즉 이 폴더의 목적은 `현재 상태를 가볍게 재현하고, raw/converted/visualization/model-code를 한 자리에서 바로 시작`할 수 있게 만드는 것입니다.
