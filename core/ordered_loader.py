"""
Ordered Data Loader for cluster-wise embedding extraction.
"""
import os
import pickle
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader


def _worker_init_fn(worker_id):
    """Set random seed for DataLoader workers."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class OrderedTrackDataset(torch.utils.data.Dataset):
    """Ordered dataset for extracting embeddings without shuffling."""

    def __init__(self, is_face, name2face_emb, name2voice_emb, data_root):
        self.is_face = is_face
        self.name2emb = name2face_emb if is_face else name2voice_emb

        name2movies_path = os.path.join(data_root, "info", "name2movies.pkl")
        with open(name2movies_path, 'rb') as f:
            name2movies = pickle.load(f)

        split_path = os.path.join(data_root, "info", "train_valid_test_names.pkl")
        with open(split_path, 'rb') as f:
            train_names = pickle.load(f)["train"]

        self.data = []
        for name in train_names:
            movies = name2movies.get(name, [])
            for movie in movies:
                movie = movie.replace("/1.6/", "/")
                filtered_keys = [k for k in self.name2emb.keys() if k.startswith(movie)]
                for k in filtered_keys:
                    self.data.append([movie, k])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        movie, short_path = self.data[index]
        inp = self.name2emb[short_path]
        return movie, inp


def get_ordered_iter(batch_size, name2face_emb, name2voice_emb, is_face, data_root):
    dataset = OrderedTrackDataset(is_face, name2face_emb, name2voice_emb, data_root)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, worker_init_fn=_worker_init_fn
    )


def extract_embeddings_core(ordered_iter, encoder, is_face):
    """
    Extract embeddings from an ordered iterator.
    Averages embeddings belonging to the same video clip.
    """
    encoder.eval()
    device = next(encoder.parameters()).device
    the_dict = collections.defaultdict(list)

    for batch_movie, input_list in ordered_iter:
        with torch.no_grad():
            input_list = [x.to(device) for x in input_list]
            if is_face:
                batch_emb = encoder.face_encoder(input_list)
            else:
                input_list = [x.float() for x in input_list]
                batch_emb = encoder.voice_encoder(input_list)

            for emb, movie in zip(batch_emb, batch_movie):
                the_dict[movie].append(emb.detach().cpu().numpy())
    encoder.train()

    # Merge by video (mean pooling)
    final_dict = {key: np.mean(arr, axis=0) for key, arr in the_dict.items()}

    videos = sorted(final_dict.keys())
    emb_array = np.array([final_dict[key] for key in videos])
    return videos, emb_array


def extract_embeddings(name2face_emb, name2voice_emb, model, data_root):
    """Extract and concatenate face & voice embeddings."""
    face_iter = get_ordered_iter(512, name2face_emb, name2voice_emb, True, data_root)
    movies, emb_face = extract_embeddings_core(face_iter, model, is_face=True)

    voice_iter = get_ordered_iter(512, name2face_emb, name2voice_emb, False, data_root)
    movies2, emb_voice = extract_embeddings_core(voice_iter, model, is_face=False)

    assert len(movies2) == len(movies)
    final_emb = np.hstack([emb_voice, emb_face])
    return movies, final_emb, emb_voice, emb_face
