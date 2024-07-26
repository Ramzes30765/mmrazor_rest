#############################################################################
"""You have to fill these args.

_base_(str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""

_base_ = '/home/dalekseenko/CV_Restaurant/cv/src/cv_api/detection/mmlab/mmyolo_rest/configs/rtmdet/rtmdet_l_phones.py'
fix_subnet = {
    "backbone.stem.0.conv_(0, 16)_16": 16,
    "backbone.stem.1.conv_(0, 16)_16": 16,
    "backbone.stem.2.conv_(0, 32)_32": 32,
    "backbone.stage1.0.conv_(0, 64)_64": 64,
    "backbone.stage1.1.short_conv.conv_(0, 32)_32": 32,
    "backbone.stage1.1.main_conv.conv_(0, 32)_32": 32,
    "backbone.stage1.1.blocks.0.conv1.conv_(0, 32)_32": 32,
    "backbone.stage1.1.final_conv.conv_(0, 64)_64": 64,
    "backbone.stage2.0.conv_(0, 128)_128": 128,
    "backbone.stage2.1.short_conv.conv_(0, 64)_64": 64,
    "backbone.stage2.1.main_conv.conv_(0, 64)_64": 64,
    "backbone.stage2.1.blocks.0.conv1.conv_(0, 64)_64": 63,
    "backbone.stage2.1.blocks.1.conv1.conv_(0, 64)_64": 64,
    "backbone.stage2.1.final_conv.conv_(0, 128)_128": 128,
    "backbone.stage3.0.conv_(0, 256)_256": 255,
    "backbone.stage3.1.short_conv.conv_(0, 128)_128": 128,
    "backbone.stage3.1.main_conv.conv_(0, 128)_128": 128,
    "backbone.stage3.1.blocks.0.conv1.conv_(0, 128)_128": 127,
    "backbone.stage3.1.blocks.1.conv1.conv_(0, 128)_128": 126,
    "backbone.stage3.1.final_conv.conv_(0, 256)_256": 256,
    "backbone.stage4.0.conv_(0, 512)_512": 380,
    "backbone.stage4.1.conv1.conv_(0, 256)_256": 256,
    "backbone.stage4.1.conv2.conv_(0, 512)_512": 483,
    "backbone.stage4.2.short_conv.conv_(0, 256)_256": 181,
    "backbone.stage4.2.main_conv.conv_(0, 256)_256": 144,
    "backbone.stage4.2.blocks.0.conv1.conv_(0, 256)_256": 142,
    "backbone.stage4.2.blocks.0.conv2.pointwise_conv.conv_(0, 256)_256": 189,
    "backbone.stage4.2.final_conv.conv_(0, 512)_512": 195,
    "neck.reduce_layers.2.conv_(0, 256)_256": 256,
    "neck.top_down_layers.0.0.short_conv.conv_(0, 128)_128": 128,
    "neck.top_down_layers.0.0.main_conv.conv_(0, 128)_128": 128,
    "neck.top_down_layers.0.0.blocks.0.conv1.conv_(0, 128)_128": 128,
    "neck.top_down_layers.0.0.blocks.0.conv2.pointwise_conv.conv_(0, 128)_128": 128,
    "neck.top_down_layers.0.0.final_conv.conv_(0, 256)_256": 254,
    "neck.top_down_layers.0.1.conv_(0, 128)_128": 128,
    "neck.top_down_layers.1.short_conv.conv_(0, 64)_64": 64,
    "neck.top_down_layers.1.main_conv.conv_(0, 64)_64": 64,
    "neck.top_down_layers.1.blocks.0.conv1.conv_(0, 64)_64": 64,
    "neck.top_down_layers.1.blocks.0.conv2.pointwise_conv.conv_(0, 64)_64": 64,
    "neck.top_down_layers.1.final_conv.conv_(0, 128)_128": 128,
    "neck.downsample_layers.0.conv_(0, 128)_128": 127,
    "neck.bottom_up_layers.0.short_conv.conv_(0, 128)_128": 127,
    "neck.bottom_up_layers.0.main_conv.conv_(0, 128)_128": 98,
    "neck.bottom_up_layers.0.blocks.0.conv1.conv_(0, 128)_128": 89,
    "neck.bottom_up_layers.0.blocks.0.conv2.pointwise_conv.conv_(0, 128)_128": 128,
    "neck.bottom_up_layers.0.final_conv.conv_(0, 256)_256": 206,
    "neck.downsample_layers.1.conv_(0, 256)_256": 252,
    "neck.bottom_up_layers.1.short_conv.conv_(0, 256)_256": 239,
    "neck.bottom_up_layers.1.main_conv.conv_(0, 256)_256": 40,
    "neck.bottom_up_layers.1.blocks.0.conv1.conv_(0, 256)_256": 50,
    "neck.bottom_up_layers.1.blocks.0.conv2.pointwise_conv.conv_(0, 256)_256": 241,
    "neck.bottom_up_layers.1.final_conv.conv_(0, 512)_512": 354
}
divisor = 16

##############################################################################

architecture = _base_.model

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherDeploySubModel',
    architecture=architecture,
    fix_subnet=fix_subnet,
    divisor=divisor,
)
