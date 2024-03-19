# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import nncore
import torch
from nncore.dataset import DATASETS
from nncore.parallel import DataContainer
from torch.utils.data import Dataset

from .utils import eval_qvhighlights, save_json
import json


@DATASETS.register()
class QVHighlights(Dataset):

    def __init__(self, label_path, video_path, audio_path, query_path):
        self.label = nncore.load(label_path)

        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.query_path = query_path

        # self.clip_model_512, _ =  clip.load(r'../../data/mydata/ViT-B-32.pt', device=torch.device('cpu'))
        # visual = self.clip_model_512.visual
        # self.ln_post = visual.ln_post
        # self.proj = visual.proj
        #
        # del self.clip_model_512, _, visual


        self.data_num = 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        #################################

        # inner_feature, video = self.my_get_video1(idx)
        inner_feature, video = self.my_get_video2(idx)
        # video = self.get_video(idx)
        #################################
        audio = self.get_audio(idx)
        query = self.get_query(idx)

        saliency = self.get_saliency(idx)
        boundary = self.get_boundary(idx)

        if saliency is None:
            num_clips = min(c.size(0) for c in (video, audio))
            saliency = torch.ones(num_clips)
        else:
            num_clips = min(c.size(0) for c in (video, audio, saliency))
            saliency = saliency[:num_clips]

        #################################
        hidden_len = len(inner_feature)
        new_inner_feature = torch.zeros(hidden_len, num_clips, 768)
        for i in range(hidden_len):
            new_inner_feature[i] = torch.as_tensor(inner_feature[i][:num_clips])

        #################################

        data = dict(
            video=DataContainer(video[:num_clips]),
            audio=DataContainer(audio[:num_clips]),
            query=DataContainer(query, pad_value=float('inf')),
            saliency=DataContainer(saliency, pad_value=-1),
            meta=DataContainer(self.label[idx], cpu_only=True))

        if boundary is not None:
            data['boundary'] = DataContainer(boundary, pad_value=-1)

        data['inner_feature'] = DataContainer(new_inner_feature.float())

        return data

    def my_get_video1(self, idx):
        '''
        读n层的数据，和最后一层叠成一个list，维度为75, 50, 768 * n, 75, 512
        :param idx:
        :return:
        '''
        vid = self.label[idx]['vid']
        u_path = '/media/my/新加卷/mydata/'
        video_path = ['hero_features_3090_12']
        inner_feature = [torch.Tensor(nncore.load(nncore.join(u_path, path, vid + '.npz'))['last_hidden_state']) for path in video_path]
        frames = torch.Tensor(nncore.load(nncore.join(self.video_path[0], f'{vid}.npz'))['pooler_output'])

        # 3, frames_num, 50, 768; frames_num, 512
        return inner_feature, frames.float()

    def my_get_video2(self, idx):
        '''
        读n个层的数据，和最后一层叠成一个list，维度为75, 768 * n, 75, 512
        :param idx:
        :return:
        '''
        vid = self.label[idx]['vid']
        u_path = '/media/my/新加卷/mydata/'
        video_path = ['hero_features_3090_9_h']
        inner_feature = [torch.Tensor(nncore.load(nncore.join(u_path, path, vid + '.npz'))['last_hidden_state']) for path in video_path]

        # frames = torch.Tensor(nncore.load(nncore.join(self.video_path[0], f'{vid}.npz'))['pooler_output'])
        #
        # return inner_feature, frames.float()

        frame0 = torch.Tensor(nncore.load(nncore.join(self.video_path[0], f'{vid}.npz'))['features'])
        frame1 = torch.Tensor(nncore.load(nncore.join(self.video_path[1], f'{vid}.npz'))['pooler_output'])
        num_clips = min(frame0.shape[0], frame1.shape[0])
        frames = torch.cat((frame0[:num_clips], frame1[:num_clips]), dim=1)

        # n, frames_num, 768; frames_num, 512
        return inner_feature, frames.float()

    def get_video(self, idx):
        vid = self.label[idx]['vid']
        # 单个的
        # video = [
        #     nncore.load(nncore.join(path, f'{vid}.npz'))['pooler_output']
        #     for path in self.video_path
        # ]
        # num_clips = video[0].shape[0]

        # sf + clip
        video = []
        video.append(nncore.load(nncore.join(self.video_path[0], f'{vid}.npz'))['features'])
        video.append(nncore.load(nncore.join(self.video_path[1], f'{vid}.npz'))['pooler_output'])
        num_clips = min(video[0].shape[0], video[1].shape[0])

        video = [torch.from_numpy(v[:num_clips]) for v in video]

        return torch.cat(video, dim=1).float()

    def get_audio(self, idx):
        vid = self.label[idx]['vid']
        audio = nncore.load(nncore.join(self.audio_path, f'{vid}.npy'))
        return torch.from_numpy(audio).float()

    def get_query(self, idx):
        qid = self.label[idx]['qid']
        query = nncore.load(nncore.join(self.query_path, f'qid{qid}.npz'))
        return torch.from_numpy(query['last_hidden_state']).float()

    def get_saliency(self, idx):
        if 'saliency_scores' in self.label[idx]:
            saliency = [0] * int(self.label[idx]['duration'] / 2)
            for clip_id, score in zip(self.label[idx]['relevant_clip_ids'],
                                      self.label[idx]['saliency_scores']):
                saliency[clip_id] = sum(score) / 12
            return torch.Tensor(saliency)

    def get_boundary(self, idx):
        if 'relevant_windows' in self.label[idx]:
            return torch.Tensor(self.label[idx]['relevant_windows'])

    def evaluate(self, blob, **kwargs):
        num_samples, collected = len(blob), []
        blob = nncore.to_dict_of_list(blob)

        for i in range(num_samples):
            pred = dict(
                qid=blob['meta'][i][0]['qid'], vid=blob['meta'][i][0]['vid'])

            if 'saliency' in blob:
                pred['pred_saliency_scores'] = blob['saliency'][i][0].tolist()

            if 'boundary' in blob:
                pred['pred_relevant_windows'] = blob['boundary'][i][0].tolist()

            collected.append(pred)

        # with open("hddata.json", "w") as f:
        #     json.dump(collected, f)
        label = nncore.load(self.label_path)
        save_json(collected, label)

        results = eval_qvhighlights(collected, label)['brief']

        return results
