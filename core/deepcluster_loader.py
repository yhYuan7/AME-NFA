"""
DeepCluster Training Loader with Adaptive Sample Weighting.
"""
import os
import pickle
import random
import torch
from torch.utils.data import DataLoader


def _worker_init_fn(worker_id):
    import numpy as np
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class DeepClusterDataset(torch.utils.data.Dataset):
    """
    Dataset for DeepCluster stage with pseudo-labels and confidence weights.
    """

    def __init__(self, data_root, full_length, name2face_emb, name2voice_emb,
                 movie2label, movie2weight):
        self.data_root = data_root
        self.full_length = full_length
        self.name2face_emb = name2face_emb
        self.name2voice_emb = name2voice_emb
        self.movie2label = movie2label
        self.movie2weight = movie2weight

        self.train_movie_list = list(movie2label.keys())

        # Build movie-to-path mappings
        self.movie2jpg_path = {}
        self.movie2wav_path = {}
        name2movies_path = os.path.join(data_root, "info", "name2movies.pkl")
        with open(name2movies_path, 'rb') as f:
            name2movies = pickle.load(f)

        for name, movie_list in name2movies.items():
            for movie_obj in movie_list:
                movie_name = movie_obj.replace("/1.6/", "/")
                filtered_keys = [k for k in name2face_emb.keys() if k.startswith(movie_name)]
                self.movie2jpg_path[movie_name] = filtered_keys.copy()
                self.movie2wav_path[movie_name] = filtered_keys.copy()

    def __len__(self):
        return self.full_length

    def __getitem__(self, index):
        movie = self.train_movie_list[index % len(self.train_movie_list)]
        label = self.movie2label[movie]
        weight = self.movie2weight.get(movie, 1.0)

        img = random.choice(self.movie2jpg_path.get(movie, [""]))
        wav = random.choice(self.movie2wav_path.get(movie, [""]))

        emb_wav = self.name2voice_emb[wav]
        emb_face = self.name2face_emb[img]

        wav_tensors = [
            torch.as_tensor(emb_wav[0], dtype=torch.float32),
            torch.as_tensor(emb_wav[1], dtype=torch.float32)
        ]
        face_tensors = [
            torch.as_tensor(arr, dtype=torch.float32) for arr in emb_face
        ]

        return wav_tensors, face_tensors, torch.LongTensor([label]), torch.FloatTensor([weight])


def get_deepcluster_iter(batch_size, full_length, name2face_emb, name2voice_emb,
                         movie2label, movie2weight, data_root):
    dataset = DeepClusterDataset(
        data_root, full_length, name2face_emb, name2voice_emb,
        movie2label, movie2weight
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, worker_init_fn=_worker_init_fn
    )
