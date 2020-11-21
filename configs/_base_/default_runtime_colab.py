checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
work_dirs = osp.join('/content/drive/MyDrive/ts/', osp.splitext(osp.basename(args.config))[0])
workflow = [('train', 1)]
