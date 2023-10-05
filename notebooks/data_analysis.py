import sys, os
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(1, sys.path[0] + '/..')
from src.misc.files import read_image


df = pd.DataFrame(columns=['min', 'max', 'mean', 'std', 'median'])

img_dir = '../data/raw/val/images/'
for i, img_path in tqdm(enumerate(os.listdir(img_dir))):
    img_path = os.path.join(img_dir, img_path)
    img, _ = read_image(img_path)
    metrics = np.min(img), np.max(img), np.mean(img), np.std(img), np.median(img)

    # add metrics to dataframe
    df.loc[i] = metrics

# save dataframe to csv
df.to_csv('data_summary_val.csv')
