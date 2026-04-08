from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DepthWeightSchedulerHook(Hook):
    """2단계 Depth 학습 전략 Hook.

    Stage 1 (epoch < switch_epoch): code_weight의 depth 항목을 낮게 유지 (0.2).
    Stage 2 (epoch >= switch_epoch): depth 가중치를 stage2_weight(1.0)으로 전환.

    Args:
        switch_epoch (int): Stage 2로 전환할 epoch (0-indexed).
            예: switch_epoch=6 → epoch 7부터 Stage 2 적용.
        depth_index (int): code_weight 리스트에서 depth에 해당하는 인덱스. 기본값 2.
        stage2_weight (float): Stage 2에서 사용할 depth 가중치. 기본값 1.0.
    """

    def __init__(
        self,
        switch_epoch: int = 6,
        depth_index: int = 2,
        stage2_weight: float = 1.0,
    ) -> None:
        self.switch_epoch = switch_epoch
        self.depth_index = depth_index
        self.stage2_weight = stage2_weight
        self._switched = False

    def before_train_epoch(self, runner) -> None:
        if self._switched:
            return
        if runner.epoch >= self.switch_epoch:
            model = runner.model
            head = getattr(model, 'bbox_head', None)
            if head is None and hasattr(model, 'module'):
                head = getattr(model.module, 'bbox_head', None)

            if head is not None and head.train_cfg is not None:
                code_weight = head.train_cfg.get('code_weight', None)
                if code_weight is not None:
                    old_val = code_weight[self.depth_index]
                    code_weight[self.depth_index] = self.stage2_weight
                    runner.logger.info(
                        f'[DepthWeightScheduler] Epoch {runner.epoch + 1}: '
                        f'depth code_weight {old_val:.1f} → {self.stage2_weight:.1f} '
                        f'(Stage 2 시작)'
                    )
                    self._switched = True
                else:
                    runner.logger.warning(
                        '[DepthWeightScheduler] code_weight를 train_cfg에서 찾을 수 없습니다.'
                    )
            else:
                runner.logger.warning(
                    '[DepthWeightScheduler] bbox_head 또는 train_cfg를 찾을 수 없습니다.'
                )
