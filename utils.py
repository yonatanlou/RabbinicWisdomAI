import numpy as np
import random
import torch


def split_data(df, random_split=False, train_size=0.8):
    if random_split:
        train_idxs = random.sample(list(df.index), int(train_size * df.shape[0]))
        df_train = df[df.index.isin(train_idxs)]
        df_test = df[~df.index.isin(train_idxs)]
    else:
        train_size = int(train_size * df.shape[0])
        df_train = df.iloc[:train_size,]
        df_test = df.iloc[train_size:,]

    return df_train, df_test


def set_seed_globally(seed=42):
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
