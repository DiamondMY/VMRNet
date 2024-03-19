_base_ = 'datasets'
# dataset settings
dataset_type = 'QVHighlights'
data_root = '../../data/mydata/myqvhighlights/'
u_path = '/media/my/新加卷/mydata/'
# data_root + 'slowfast_features'

data = dict(
    train=dict(
        type=dataset_type,
        label_path=data_root + 'highlight_train_release.jsonl',
        video_path=[
            u_path + 'slowfast_features', u_path + 'hero_features_3090'
        ],
        audio_path=data_root + 'pann_features',
        query_path=u_path + 'new_clip_text',
        loader=dict(batch_size=32, num_workers=4, shuffle=True)),
    val=dict(
        type=dataset_type,
        label_path=data_root + 'highlight_val_release.jsonl',
        video_path=[
            u_path + 'slowfast_features', u_path + 'hero_features_3090'
        ],
        audio_path=data_root + 'pann_features',
        query_path=u_path + 'new_clip_text',
        loader=dict(batch_size=1, num_workers=4, shuffle=False)))
