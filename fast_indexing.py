import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
df, md, dfidx = load_data(load_only_train=False)
generator_demo(df, md, dfidx)
'''

# df, md, dfidx = load_data()
def load_data(load_only_train=True):
    dtype_dict = {'object_id': np.int32, 'mjd': np.float32, 'passband': np.int8, 'flux': np.float32,
                  'flux_err': np.float32,
                  'detected': bool}

    if load_only_train:
        df = pd.read_csv('training_set.csv', dtype=dtype_dict)
        md = pd.read_csv('training_set_metadata.csv')
    else:
        df = pd.read_csv('test_set.csv.zip', dtype=dtype_dict)
        df['train_set'] = False
        df = df.append(pd.read_csv('training_set.csv', dtype=dtype_dict), ignore_index=True)
        df['train_set'].fillna(True, inplace=True)

        md = pd.read_csv('test_set_metadata.csv')
        md['train_set'] = False
        md = md.append(pd.read_csv('training_set_metadata.csv'), ignore_index=True)
        md['train_set'].fillna(True, inplace=True)


    mdcols = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod',
              'mwebv']
    for c in mdcols:
        md[c] = md[c] / md[c].max()
        md[c].fillna(-1.0, inplace=True)

    df.reset_index(inplace=True)

    #dfidx contains the index of the first and last data point for each object_id
    dfidx = df.index.to_series().groupby(df['object_id']).agg(['first', 'last'])
    df.drop(columns=['index'], inplace=True)

    df.set_index('object_id', inplace=True)
    md.set_index('object_id', inplace=True)

    return df, md, dfidx



def process_item(q, md, df_cols, n_time_steps):
    X = np.zeros((1, n_time_steps, 4)) #
    use_passband = 3

    t = q[q['passband'] == use_passband]
    t.reset_index(inplace=True)

    idx = list(t.index[0:n_time_steps])

    X[0, 0:len(idx), :] = t.loc[idx, df_cols].values

    #for cases where there are fewer than n_time_steps samples, repeat the last sample
    if len(idx) > 1:
        X[0, len(idx):, :] = X[0, len(idx) - 1, :]

    return (X, md.values)


def gen_data2(df, dfidx, md, n_time_steps, md_use, n_bands, batch_size):
    df_cols = ['mjd', 'passband', 'flux', 'flux_err']

    object_ids = list(dfidx.index)

    np.random.shuffle(object_ids)
    n_samples = len(object_ids)
    j = 0

    while True:
        if j + batch_size > n_samples:
            np.random.shuffle(object_ids)
            j = 0

        use_ids = object_ids[j:j + batch_size]
        j += batch_size

        d = [process_item(df.iloc[dfidx.loc[use_id, 'first']:dfidx.loc[use_id, 'last'] + 1], md.loc[use_id, md_use],
                          df_cols, n_time_steps) for use_id in use_ids]
        X = np.vstack([d[i][0] for i in range(batch_size)])
        Xmd = np.vstack([d[i][1] for i in range(batch_size)])

        yield [X, Xmd]



def generator_demo(df, md, dfidx):
    batch_size = 64
    n_time_steps = 32
    n_bands = df['passband'].nunique()

    md_use = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv']


    train_gen = gen_data2(df, dfidx, md, n_time_steps, md_use, n_bands, batch_size)

    sample = next(train_gen)

    plt.close('all')

    plt.plot(sample[0][0, :, 0], sample[0][0, :, 2], 'k.-')
    plt.plot(sample[0][1, :, 0], sample[0][1, :, 2], 'r.-')
    plt.plot(sample[0][2, :, 0], sample[0][2, :, 2], 'b.-')

    plt.show()

