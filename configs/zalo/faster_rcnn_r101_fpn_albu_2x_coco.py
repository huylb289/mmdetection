_base_ = './faster_rcnn_r50_fpn_albu_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
