#############################################################################
"""# You have to fill these args.

_base_(str): The path to your pruning config file.
pruned_path (str): The path to the checkpoint of the pruned model.
finetune_lr (float): The lr rate to finetune. Usually, we directly use the lr
    rate of the pretrain.
"""

_base_ = '/home/dalekseenko/CV_Restaurant/cv/src/cv_api/detection/mmlab/mmrazor_rest/configs/pruning/mmdet/group_fisher/rtmdet/group_fisher_act_prune_rtmdet_fpn_1x_coco.py'
pruned_path = '/home/dalekseenko/CV_Restaurant/cv/src/cv_api/detection/mmlab/work_dirs/group_fisher_act_prune_rtmdet_fpn_1x_coco/best_coco_bbox_mAP_epoch_292.pth'  # noqa
finetune_lr = 0.005
##############################################################################
algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)

# restore lr
optim_wrapper = dict(optimizer=dict(lr=finetune_lr))

# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]

# delete ddp
model_wrapper_cfg = None
