dataset_type = 'HRSCDataset'
data_root = '/home/z/code/kch/RS_Dataset/hrsc2016/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RotatedResize', img_scale=(800, 512), keep_ratio=True),
    dict(type='RotatedRandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', rate=0.5, angles=[30, 60, 90, 120, 150], auto_bound=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RotatedResize', keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Train/train.txt',
        img_prefix=data_root + 'Train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Test/test.txt',
        img_prefix=data_root + 'Test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Test/test.txt',
        img_prefix=data_root + 'Test/',
        pipeline=test_pipeline))
evaluation = dict(
    gt_dir='/code/kch/RS_Dataset/HRSC2016/Test/Annotations/',
    imagesetfile='/code/kch/RS_Dataset/HRSC2016/Test/test.txt')