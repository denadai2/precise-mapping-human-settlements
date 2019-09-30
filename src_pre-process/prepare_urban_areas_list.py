import numpy as np
import os
import gzip
import pandas as pd
from joblib import Parallel, delayed
from skimage import measure
import csv


def get_clusters_area(M, minpop):
    labels = measure.label(M > minpop, connectivity=2)
    regions = measure.regionprops(labels)
    cluster_pop = np.zeros(len(regions), dtype='float32')
    for i, x in enumerate(regions):
        cluster_pop[i] = x.area

    return cluster_pop


def get_clusters_km2(tile, orig_tile_km):
    if not os.path.isfile('../data/cache_numpy_05x05/{}.npz'.format(tile)):
        return None, 0
    else:
        fload = np.load('../data/cache_numpy_05x05/{}.npz'.format(tile))
        M = fload['M']

    km_per_pixel = orig_tile_km / (M.shape[0] ** 2)
    clusters = get_clusters_area(M, 0.5)

    return tile, clusters * km_per_pixel


def main():
    df_km2tiles = pd.read_csv('../data/generated_files/tiles_fullkm2_05x05.csv', dtype={'tileid': 'str'})
    df_km2tiles = df_km2tiles.set_index('tileid')
    df_km2tiles.head()

    clusters_areas_km2 = {tileid: kms for tileid, kms in Parallel(n_jobs=10, verbose=3)(
        delayed(get_clusters_km2)(tile, df_km2tiles.loc[tile, 'full_tile_km2']) for tile in df_km2tiles.index.tolist())}

    with gzip.open('../data/generated_files/filippo_areas_reduced4.csv.gz', "wt", newline="") as csvfile:
        spamwriter = csv.writer(csvfile)

        spamwriter.writerow(['tileid', 'area_km2'])
        for k, vs in clusters_areas_km2.items():
            if k:
                for v in vs:
                    spamwriter.writerow([k, v])


if __name__ == '__main__':
    main()
