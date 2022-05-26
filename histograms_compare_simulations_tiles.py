import argparse
import gzip
import json
import time
from collections import defaultdict
from itertools import chain
from operator import attrgetter
from os import listdir
from os.path import isfile
from os.path import join

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel
from joblib import delayed
from scipy.stats import energy_distance
from scipy.stats import wasserstein_distance
from skimage import measure
from tqdm import tqdm

from config_loader import *
from histogram_namedtuple import *


def filename2parameters(filename):
    name = (
        filename.replace('rybski_marco_1000x1000_s', '')
        .replace('rybski_2steps_1000x1000_s', '')
        .replace('.npz', '')
    )
    params = [float(x) for x in name.split('_')]
    return params[1], params[2], params[0]


def get_clusters_area(M):
    labels = measure.label(M, connectivity=1)
    regions = measure.regionprops(labels, cache=False)
    cluster_pop = np.zeros(len(regions), dtype='int32')
    for i, x in enumerate(regions):
        cluster_pop[i] = x.area

    return cluster_pop


def JS(p, q):
    m = (p + q) / 2
    return (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2


def get_simulation(
    bins, histogram_tile, hist1, filename, path, urbanization_perc, range_urb, tile_size=1000
):
    with np.load('{}/{}'.format(path, filename)) as fload:
        perc_urban = fload['perc']

        start = max(urbanization_perc - range_urb, 0.005)
        stop = urbanization_perc + range_urb

        max_steps = len(perc_urban)
        steps = np.arange(max_steps, dtype='int32')
        good_steps = steps[(perc_urban > start) & (perc_urban < stop)]
        saved_steps = {'s{}'.format(s): fload['s{}'.format(s)] for s in good_steps}

    comparisons = []
    perc_array = perc_urban[steps]
    for s in good_steps:
        perc_tile = perc_array[s]
        h_sim_tile = saved_steps['s{}'.format(s)]
        hist2, _ = np.histogram(np.sqrt(h_sim_tile), bins=bins)

        js_score = scipy.spatial.distance.jensenshannon(hist1, hist2)

        energy_score = energy_distance(histogram_tile, h_sim_tile)
        earth_score = wasserstein_distance(histogram_tile, h_sim_tile)

        comparisons.append(
            Comparison(
                js=float(js_score),
                energy=float(energy_score),
                earth=float(earth_score),
                filename=filename,
                purban=perc_tile,
                step=int(s),
            )
        )
    return comparisons


def process_tile(
    tileid,
    simulation_files,
    tiles_cache_folder,
    simulations_cache_folder,
    metrics,
    distances_folder,
    args,
    range_urb=0.01,
):
    with np.load('{}/{}.npz'.format(tiles_cache_folder, tileid)) as fload:
        M = fload['M']
        urban_percentage = fload['purb']

    assert 0 < urban_percentage < 1

    histogram_tile = get_clusters_area(M)
    bins = np.arange(1, 800)
    hist1, _ = np.histogram(np.sqrt(histogram_tile), bins=bins)

    filtered_simulation_files = []
    for f in simulation_files:
        params = f.split('_')
        params[-1] = params[-1].replace('.npz', '')
        params[-3] = params[-3].replace('s', '')

        if (
            args.model == '1000'
            and params[-1] != params[-2]
            and float(params[-3]) > urban_percentage + range_urb
        ):
            continue

        filtered_simulation_files.append(f)

    results = [
        get_simulation(
            bins,
            histogram_tile,
            hist1,
            filename,
            simulations_cache_folder,
            urban_percentage,
            tile_size=1000,
            range_urb=range_urb,
        )
        for filename in filtered_simulation_files
    ]
    comparisons = list(chain(*results))
    to_return = {}
    if comparisons:
        for m in metrics:
            # Comparisons
            sorted_comparisons = sorted(comparisons, key=attrgetter(m))
            sorted_comparisons_desc = sorted_comparisons[-5:][::-1]

            best_scores = [getattr(sorted_comparisons[i], m) for i in range(5)]
            worst_scores = [getattr(sorted_comparisons_desc[i], m) for i in range(5)]

            best_params = [
                filename2parameters(getattr(sorted_comparisons[i], 'filename')) for i in range(5)
            ]
            worst_params = [
                filename2parameters(getattr(sorted_comparisons_desc[i], 'filename'))
                for i in range(5)
            ]

            best_steps = [getattr(sorted_comparisons[i], 'step') for i in range(5)]
            worst_steps = [getattr(sorted_comparisons_desc[i], 'step') for i in range(5)]

            best_purban = [float(getattr(sorted_comparisons[i], 'purban')) for i in range(5)]
            worst_purban = [float(getattr(sorted_comparisons_desc[i], 'purban')) for i in range(5)]

            # Name of the file where to save the comparison
            json_filename = '{}.json'.format(tileid)
            if args.model == 'marco':
                json_filename = 'multi_{}.json'.format(tileid)
            elif args.model == 'rybski':
                json_filename = 'rybski_{}.json'.format(tileid)
            fname = '{}/{}_{}.gz'.format(distances_folder, m, json_filename)

            # Save comparisons
            dict_json = {
                'distances': [round(getattr(c, m), 6) for c in sorted_comparisons],
                'params': [filename2parameters(c.filename) for c in sorted_comparisons],
                'steps': [getattr(c, 'step') for c in sorted_comparisons],
            }

            with gzip.open(fname, 'wt', encoding="ascii") as zipfile:
                json.dump(dict_json, zipfile)

            to_return[m] = (
                tileid,
                (
                    best_scores,
                    worst_scores,
                    best_params,
                    worst_params,
                    best_steps,
                    worst_steps,
                    best_purban,
                    worst_purban,
                ),
            )

    return to_return


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Find matches between the real and simulated tiles"
    )
    parser.add_argument('--size', '-S', default=1000, type=int)
    parser.add_argument('--njobs', '-J', default=10, type=int)
    parser.add_argument('--model', '-M', default='1000', choices=['1000', 'rybski', 'marco'])

    return parser


def main():
    configs = load_config()

    parser = make_argument_parser()
    args = parser.parse_args()
    print("PARAMETERS", args)

    model_var = args.model
    if model_var == 'rybski':
        model_var = '1000'

    simulations_cache_folder = '{}/cachesimulations/{}'.format(
        configs['generated_files_path'], model_var
    )
    tiles_cache_folder = 'data/cache_numpy_05x05'
    distances_folder = '{}/simulations/distances'.format(configs['generated_files_path'])
    comparisons_folder = '{}/simulations'.format(configs['generated_files_path'])

    if len([f for f in listdir(distances_folder) if not f.startswith('.')]):
        print("WARNING: distances folder is not empty")

    metrics = ['js', 'energy', 'earth']
    simulation_files = [
        f
        for f in listdir(simulations_cache_folder)
        if isfile(join(simulations_cache_folder, f)) and 'npz' in f
    ]

    new_simulation_files = []
    for s in simulation_files:
        s_splitted = s.split('_')
        if s_splitted[-1].replace('.npz', '') != s_splitted[-2] or (
            s_splitted[-1].replace('.npz', '') == s_splitted[-2] and s_splitted[-3] == 's0.5'
        ):
            new_simulation_files.append(s)
    simulation_files = new_simulation_files
    if args.model == 'marco':
        simulation_files = [
            f
            for f in listdir(simulations_cache_folder)
            if isfile(join(simulations_cache_folder, f)) and 'npz' in f and 'marco' in f
        ]
    elif args.model == 'rybski':
        filename2params = lambda x: (
            float(x.split('_')[-3][1:]),
            float(x.split('_')[-2]),
            float(x.split('_')[-1][:-4]),
        )
        simulation_files = [
            f
            for f in listdir(simulations_cache_folder)
            if isfile(join(simulations_cache_folder, f))
            and 'npz' in f
            and filename2params(f)[1] == filename2params(f)[2]
        ]

    print("N Simulations", len(simulation_files))

    df_classes = pd.read_csv(
        '{}/quantiles_classes.csv'.format(configs['generated_files_path']), dtype={'tileid': str}
    )

    # Read summary of tiles
    df = pd.read_csv(
        '{}/summary_tiles_05x05.csv'.format(configs['generated_files_path']),
        dtype={'tileid': 'str'},
    )
    df['perc_urban'] = df['urban_area_km2'] / df['original_km2']
    df['perc_constraint'] = 1 - (df['tile_km2'] / df['original_km2'])

    df = pd.merge(df[['tileid', 'perc_urban', 'perc_constraint']], df_classes, on='tileid')

    # Creates the bins
    lb, hb = (0.01, 1.0)

    df_sampled = df[(df['perc_urban'] > lb) & (df['perc_urban'] <= hb)]
    print("SAMPLE size ({}, {}]: {}".format(lb, hb, len(df_sampled)))

    tiles = df_sampled['tileid'].values
    parallel_results = [
        r
        for r in Parallel(n_jobs=args.njobs)(
            delayed(process_tile)(
                tileid,
                simulation_files,
                tiles_cache_folder,
                simulations_cache_folder,
                metrics,
                distances_folder,
                args,
                range_urb=0.005,
            )
            for _, tileid in enumerate(tqdm(tiles))
        )
        if r
    ]

    similarity_results = defaultdict(dict)
    for p in parallel_results:
        for m, (tileid, comparisons) in p.items():
            similarity_results[m][tileid] = comparisons

    for m in metrics:
        with open(
            '{folder}/{score}{model}_{d}x{d}_lb{lb}_hb{hb}_all_1000.json'.format(
                folder=comparisons_folder,
                score=m,
                d=args.size,
                lb=lb,
                hb=hb,
                model=(model_var if args.model != 'rybski' else 'rybski'),
            ),
            'w',
        ) as fp:
            json.dump(similarity_results[m], fp)


if __name__ == '__main__':
    main()
