"""
Cross-Modal Evaluation Protocols: Verification, Retrieval, Matching, 1:N.
"""
import os
import pickle
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, features, data_root):
        self.data = data
        self.features = features
        self.data_root = data_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        track = self.data[index]
        if ".wav" in track:
            tmp = track.split("/")
            clip = int(tmp[-1].replace(".wav", ""))
            track_path = os.path.join(
                self.data_root, "features",
                "%s/%s/%s/compact.pkl" % (tmp[0], tmp[1], clip)
            )
        else:
            track_path = os.path.join(
                self.data_root, "features",
                track.replace("/1.6/", "/") + "/compact.pkl"
            )

        with open(track_path, 'rb') as f:
            tmp_dict = pickle.load(f)

        all_features = [torch.FloatTensor(tmp_dict[f]) for f in self.features]
        return all_features, self.data[index]


def _get_track2emb(encoder, path_set, features, data_root):
    """Generate path-to-embedding dictionary."""
    loader = DataLoader(
        EvalDataset(list(path_set), features, data_root),
        batch_size=512, shuffle=False, pin_memory=True
    )
    p2emb = {}
    with torch.no_grad():
        for data, key_list in loader:
            data = [x.cuda() for x in data]
            emb_batch = encoder(data).detach().cpu().numpy()
            for key, emb in zip(key_list, emb_batch):
                p2emb[key] = emb
    return p2emb


# ==================== Metric Core Functions ====================

def calc_verification(pair_list, v2emb, f2emb):
    """Compute verification AUC from (wav, face, label) pairs."""
    scores, labels = [], []
    for wav, face, label in pair_list:
        if wav not in v2emb or face not in f2emb:
            continue
        v, f = v2emb[wav], f2emb[face]
        sim = np.dot(v, f) / (np.linalg.norm(v) * np.linalg.norm(f) + 1e-12)
        scores.append(sim)
        labels.append(label)
    if len(labels) == 0:
        return 0.0
    return float(roc_auc_score(labels, scores))


def calc_map_value(retrieval_lists, v2emb, f2emb):
    """
    Compute mean Average Precision for cross-modal retrieval.
    retrieval_lists: dict with keys 'v2f' and/or 'f2v'.
    Each value is a dict mapping query -> [gallery_keys] (first is ground truth).
    """
    def _mean_ap(queries, q2emb, g2emb):
        aps = []
        for q_key, candidates in queries.items():
            if q_key not in q2emb:
                continue
            cands = [c for c in candidates if c in g2emb]
            if len(cands) == 0:
                continue
            q_emb = q2emb[q_key]
            g_embs = np.array([g2emb[c] for c in cands])
            sims = g_embs @ q_emb
            # First candidate is assumed ground truth
            rels = np.zeros(len(cands))
            rels[0] = 1
            # AP calculation
            order = np.argsort(-sims)
            rels_sorted = rels[order]
            precisions = np.cumsum(rels_sorted) / (np.arange(len(rels_sorted)) + 1)
            ap = np.sum(precisions * rels_sorted) / max(1, np.sum(rels_sorted))
            aps.append(ap)
        return np.mean(aps) if aps else 0.0

    map_vf, map_fv = 0.0, 0.0
    if isinstance(retrieval_lists, dict):
        if "v2f" in retrieval_lists:
            map_vf = _mean_ap(retrieval_lists["v2f"], v2emb, f2emb)
        if "f2v" in retrieval_lists:
            map_fv = _mean_ap(retrieval_lists["f2v"], f2emb, v2emb)
    return map_vf, map_fv


def calc_matching(match_list, v2emb, f2emb):
    """
    Compute matching score (Top-1 accuracy).
    match_list: list of (wav, face, label).
    Returns (ms_v2f, ms_f2v).
    """
    # v2f: for each voice, find closest face
    v_queries = collections.defaultdict(list)
    f_queries = collections.defaultdict(list)
    for wav, face, _ in match_list:
        v_queries[wav].append(face)
        f_queries[face].append(wav)

    def _top1_acc(queries, q2emb, g2emb):
        correct, total = 0, 0
        for q_key, candidates in queries.items():
            if q_key not in q2emb:
                continue
            cands = [c for c in candidates if c in g2emb]
            if len(cands) == 0:
                continue
            q_emb = q2emb[q_key]
            g_embs = np.array([g2emb[c] for c in cands])
            sims = g_embs @ q_emb
            pred = cands[np.argmax(sims)]
            # Assume first candidate in original list is GT (adapt if needed)
            gt = candidates[0]
            correct += int(pred == gt)
            total += 1
        return correct / max(1, total)

    import collections
    ms_vf = _top1_acc(v_queries, v2emb, f2emb)
    ms_fv = _top1_acc(f_queries, f2emb, v2emb)
    return ms_vf, ms_fv


def handle_1_n(match_list, is_v2f, key2emb):
    """
    1:N matching accuracy.
    match_list: list of (query, [candidates]).
    """
    correct, total = 0, 0
    for query, candidates in match_list:
        if query not in key2emb:
            continue
        cands = [c for c in candidates if c in key2emb]
        if len(cands) == 0:
            continue
        q_emb = key2emb[query]
        g_embs = np.array([key2emb[c] for c in cands])
        sims = g_embs @ q_emb
        pred = cands[np.argmax(sims)]
        gt = candidates[0]  # first is ground truth
        correct += int(pred == gt)
        total += 1
    acc = correct / max(1, total)
    return {"rank1_acc": acc, "correct": correct, "total": total}


# ==================== Evaluator Class ====================

class CrossModalEvaluator:
    def __init__(self, features, data_root):
        self.face_features = [f for f in features if f.startswith("f_")]
        self.voice_features = [f for f in features if f.startswith("v_")]
        self.data_root = data_root

    def _to_emb_dict(self, model, jpg_set, wav_set):
        model.eval()
        f2emb = _get_track2emb(model.face_encoder, jpg_set, self.face_features, self.data_root)
        v2emb = _get_track2emb(model.voice_encoder, wav_set, self.voice_features, self.data_root)
        model.train()
        return v2emb, f2emb

    def do_valid(self, model):
        pkl_path = os.path.join(self.data_root, "evals", "valid_verification.pkl")
        data = self._load_pkl(pkl_path)
        v2emb, f2emb = self._to_emb_dict(model, data["jpg_set"], data["wav_set"])
        return {"valid/auc": calc_verification(data["list"], v2emb, f2emb)}

    def do_test(self, model):
        obj = {}
        # Verification
        obj["test/auc"] = self._run_verification(model, "test_verification.pkl")
        obj["test/auc_g"] = self._run_verification(model, "test_verification_g.pkl")
        obj["test/auc_n"] = self._run_verification(model, "test_verification_n.pkl")

        # Retrieval
        pkl_path = os.path.join(self.data_root, "evals", "test_retrieval.pkl")
        data = self._load_pkl(pkl_path)
        v2emb, f2emb = self._to_emb_dict(model, data["jpg_set"], data["wav_set"])
        obj["test/map_v2f"], obj["test/map_f2v"] = calc_map_value(data["retrieval_lists"], v2emb, f2emb)

        # Matching
        pkl_path = os.path.join(self.data_root, "evals", "test_matching.pkl")
        data = self._load_pkl(pkl_path)
        v2emb, f2emb = self._to_emb_dict(model, data["jpg_set"], data["wav_set"])
        obj["test/ms_v2f"], obj["test/ms_f2v"] = calc_matching(data["match_list"], v2emb, f2emb)

        pkl_path = os.path.join(self.data_root, "evals", "test_matching_g.pkl")
        data = self._load_pkl(pkl_path)
        v2emb, f2emb = self._to_emb_dict(model, data["jpg_set"], data["wav_set"])
        obj["test/ms_v2f_g"], obj["test/ms_f2v_g"] = calc_matching(data["match_list"], v2emb, f2emb)

        # 1:N Matching
        pkl_path = os.path.join(self.data_root, "evals", "test_matching_1N.pkl")
        data = self._load_pkl(pkl_path)
        v2emb, f2emb = self._to_emb_dict(model, data["jpg_set"], data["wav_set"])
        key2emb = {**v2emb, **f2emb}
        v2f_res = handle_1_n(data["match_list"], is_v2f=True, key2emb=key2emb)
        f2v_res = handle_1_n(data["match_list"], is_v2f=False, key2emb=key2emb)
        obj.update({f"test/1n_{k}": v for k, v in {**v2f_res, **f2v_res}.items()})
        return obj

    def _run_verification(self, model, filename):
        pkl_path = os.path.join(self.data_root, "evals", filename)
        data = self._load_pkl(pkl_path)
        v2emb, f2emb = self._to_emb_dict(model, data["jpg_set"], data["wav_set"])
        return calc_verification(data["list"], v2emb, f2emb)

    def _load_pkl(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def do_retrieval_samples(self, model, k=10):
        """Demo: print top-k retrieval results for random samples."""
        pkl_path = os.path.join(self.data_root, "evals", "test_retrieval.pkl")
        data = self._load_pkl(pkl_path)
        v2emb, f2emb = self._to_emb_dict(model, data["jpg_set"], data["wav_set"])

        random_f = random.choice(list(f2emb.keys()))
        random_v = random.choice(list(v2emb.keys()))

        def _topk(query_emb, candidates_emb, k):
            dists = {key: np.linalg.norm(query_emb - emb) for key, emb in candidates_emb.items()}
            return [x[0] for x in sorted(dists.items(), key=lambda x: x[1])[:k]]

        top_v2f = _topk(v2emb[random_v], f2emb, k)
        top_f2v = _topk(f2emb[random_f], v2emb, k)

        print(f"Voice query: {random_v}")
        for i, m in enumerate(top_v2f, 1):
            print(f"  {i}. {m}")
        print(f"Face query: {random_f}")
        for i, m in enumerate(top_f2v, 1):
            print(f"  {i}. {m}")

        map_vf, map_fv = calc_map_value(data["retrieval_lists"], v2emb, f2emb)
        return map_vf, map_fv
