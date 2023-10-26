import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sys, os

sys.path.insert(1, sys.path[0] + '/..')
from src.postprocess import _post_process
from src.data.utils import simplify_labels
from src.misc.post import calculate_metrics


def gridsearch(data, l1 = 101, l2 = 101, n1 = 0, n2 = 0):
    ranges = {
        'prob_thresh' : np.linspace(0, 1, l1-1, endpoint = True),
        'bone_thresh' : np.array([200]),
        'size_thresh' : np.linspace(0, 3000, l2, endpoint = True),
        'prob_thresh_log' : np.logspace(-6, 0, l1-1, endpoint = True),
    }

    combis = []
    combis_log = []
    for i, x in enumerate(ranges['prob_thresh']):
            for k, z in enumerate(ranges['size_thresh']):
                if (i + k) < (l1 - n1 + l2 - n2):
                    combis.append(np.array((ranges['prob_thresh'][i], ranges['bone_thresh'][0], ranges['size_thresh'][k])))
                    combis_log.append(np.array((ranges['prob_thresh_log'][i], ranges['bone_thresh'][0], ranges['size_thresh'][k])))
    combis = np.array(combis)
    combis_log = np.array(combis_log)

    gs = pd.DataFrame(columns = list(ranges.keys())[:3], data = combis)
    gs_log = pd.DataFrame(columns = list(ranges.keys())[:3], data = combis_log)

    for metric in ['accuracy', 'precision', 'recall']:
        gs[metric] = [[] for _ in range(len(gs))]
        gs_log[metric] = [[] for _ in range(len(gs))]

    p_nonzero = np.array([])
    device = 'cpu'

    for d in data:
        x, y, p = d['img'], d['label'], d['pred']
        p_nonzero = np.append(p_nonzero, p[p != 0])

        y = torch.tensor(simplify_labels(y), device = device, dtype = torch.int)
        for i, combi in tqdm(gs.iterrows(), total=len(gs)):
            prob_thresh, bone_thresh, size_thresh = combi[:3]

            p_post = _post_process(pred = p, image = x, prob_thresh=prob_thresh, bone_thresh=bone_thresh, size_thresh=size_thresh)
            p_post = torch.tensor(simplify_labels(p_post), device = device, dtype = torch.int)
            
            metric = calculate_metrics(y, p_post)

            for k, v in metric.items():
                gs.loc[i, k].append(v)

    gs['accuracy_mean'] = gs['accuracy'].apply(lambda x: np.mean(x))
    gs['precision_mean'] = gs['precision'].apply(lambda x: np.mean(x))
    gs['recall_mean'] = gs['recall'].apply(lambda x: np.mean(x))
    gs['accuracy_std'] = gs['accuracy'].apply(lambda x: np.std(x))
    gs['precision_std'] = gs['precision'].apply(lambda x: np.std(x))
    gs['recall_std'] = gs['recall'].apply(lambda x: np.std(x))

    return gs
