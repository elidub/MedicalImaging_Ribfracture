import sys, os
import lightning.pytorch as pl
import torch
from matplotlib import pyplot as plt
import numpy as np
import pickle
from itertools import product, combinations_with_replacement
from skimage.morphology import disk, remove_small_objects
from skimage.measure import label, regionprops
from tqdm import tqdm
import pandas as pd 
from collections import defaultdict


sys.path.insert(1, sys.path[0] + '/..')
from src.data.datamodule import DataModule
from src.data.utils import simplify_labels
from src.model.setup import setup_model
from src.misc.utils import set_seed_and_precision
from src.misc.files import SetupArgs, read_image
from src.misc.post import calculate_metrics
from src.postprocess import _remove_low_probs, _remove_spine_fp, _remove_small_objects, _post_process
from src.postprocess import main as postprocess_main, parse_option as postprocess_parse_option

from src.run import parse_option, main



def gridsearch():
    args = parse_option(notebook=True)
    args.data_dir = '../data_dev'
    args.version = 'version_sam'
    args.num_workers = 8
    device = 'cpu'

    # Predict
    args.train, args.predict = False, True
    args.splits = ['val']
    main(args)

    # Post process
    postprocess_args = postprocess_parse_option(notebook=True)
    postprocess_args.split = args.splits[0]
    postprocess_args.prediction_box_dir = os.path.join(args.log_dir, args.net, args.version, 'segmentations')
    postprocess_args.original_image_dir = args.data_dir
    postprocess_args.save_dir = os.path.join(args.log_dir, 'submissions', args.version, postprocess_args.split)

    postprocess_main(postprocess_args)


    # Setup grid search

    l = 2
    ranges = {
        'prob_thresh' : np.linspace(0, 1, l, endpoint = True),
        'bone_thresh' : np.array([200]),
        'size_thresh' : np.linspace(0, 2000, l, endpoint = True)
    }

    combis = []
    for i, x in enumerate(ranges['prob_thresh']):
            for k, z in enumerate(ranges['size_thresh']):
                if i <= k:
                    k = l-k-1
                    combis.append(np.array((ranges['prob_thresh'][i], ranges['bone_thresh'][0], ranges['size_thresh'][k])))
    combis = np.array(combis)


    # Evaluate


    img_ids = os.listdir(os.path.join(postprocess_args.prediction_box_dir, postprocess_args.split))
    gs = pd.DataFrame(columns = ranges.keys(), data = combis)

    for metric in ['accuracy', 'precision', 'recall']:
        gs[metric] = [[] for _ in range(len(gs))]

    y_pred_nonzero = np.array([])

    for img_id in img_ids:
        x, _ = read_image(os.path.join(postprocess_args.original_image_dir, 'raw', postprocess_args.split, 'images', f'{img_id}-image.nii.gz'))
        y_pred = np.load(os.path.join(postprocess_args.save_dir, f'{img_id}_pred.npy'))
        y_true, _ = read_image(os.path.join(postprocess_args.original_image_dir, 'raw', postprocess_args.split, 'labels', f'{img_id}-label.nii.gz'))
        y_true = torch.tensor(simplify_labels(y_true), device = device, dtype = torch.int)

        # y_pred_nonzero.append(y_pred[y_pred != 0])
        y_pred_nonzero = np.append(y_pred_nonzero, y_pred[y_pred != 0])
        
        for i, combi in tqdm(gs.iterrows(), total=len(gs)):
            prob_thresh, bone_thresh, size_thresh = combi[:3]

            y_pred_post = _post_process(pred = y_pred, image = x, prob_thresh=prob_thresh, bone_thresh=bone_thresh, size_thresh=size_thresh)
            y_pred_post = torch.tensor(simplify_labels(y_pred_post), device = device, dtype = torch.int)
            
            metric = calculate_metrics(y_true, y_pred_post)
            
            for k, v in metric.items():
                gs.loc[i, k].append(v)

        gs.to_csv('../store/grid_search.csv')
        np.save('../store/y_pred_nonzero.npy', y_pred_nonzero)

    gs['accuracy_mean'] = gs['accuracy'].apply(lambda x: np.mean(x))
    gs['precision_mean'] = gs['precision'].apply(lambda x: np.mean(x))
    gs['recall_mean'] = gs['recall'].apply(lambda x: np.mean(x))
    gs['accuracy_std'] = gs['accuracy'].apply(lambda x: np.std(x))
    gs['precision_std'] = gs['precision'].apply(lambda x: np.std(x))
    gs['recall_std'] = gs['recall'].apply(lambda x: np.std(x))


    gs.to_csv('../store/grid_search.csv')
    np.save('../store/y_pred_nonzero.npy', y_pred_nonzero)
    np.save('../store/combis.npy', combis)

if __name__ == '__main__':
    gridsearch()