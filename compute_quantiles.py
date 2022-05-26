import argparse
import csv
import json

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from skimage import measure
from tqdm import tqdm

from config_loader import *


def create_name_file(L, S, gamma_1, gamma_2, model_type):
    dir = L
    filename = 'rybski_2steps'
    if model_type == 'marco':
        dir = 'marco'
        filename = 'rybski_marco'

    return '{filename}_{size}x{size}_s{S}_{gamma1}_{gamma2}.npz'.format(
        size=L,
        dir=dir,
        filename=filename,
        S=int(S) if S == 1 else S,
        gamma1=gamma_1,
        gamma2=gamma_2,
    )


def sample_from_ccdf(x, y, samples=1):
    """
    x,y  :  arrays, x and y outputs of a ccdf (sorted)
    samples  :  int, number of samples to draw
    """
    n = len(x) - 1
    # To reproduce the figure
    uniform_samples = np.random.random(samples)
    indexes = np.searchsorted(1.0 - np.array(y), uniform_samples)
    return np.array([x[min(i, n)] for i in indexes])


def get_urban_phase_class(
    urban_area_km2, num_urban, tile_km2, macro_array, num_realisations=1000, num_samples=100
):
    x, y, n = macro_array

    assert urban_area_km2 > 0.0
    assert urban_area_km2 < tile_km2

    null_model_num_urb = []
    for r in range(num_realisations):
        # samples = []
        sum_samples = 0.0
        model_num_urb = 0.0
        for n in [num_samples, 1]:
            while sum_samples < urban_area_km2:
                model_num_urb += n
                new_draw = sample_from_ccdf(x, y, samples=n)
                sum_samples += sum(new_draw)
                #                 samples += list(new_draw)
                #             cumsum_samples = np.cumsum(samples)
                #             model_num_urb = np.searchsorted(cumsum_samples, urban_area_km2 )

            proposed_fixed = sum_samples - sum(new_draw)
            if n != 1 or (
                proposed_fixed > 0
                and abs(urban_area_km2 - proposed_fixed) < abs(urban_area_km2 - sum_samples)
            ):
                sum_samples = proposed_fixed
                model_num_urb -= n
        model_num_urb += np.searchsorted(np.cumsum(new_draw), urban_area_km2 - sum_samples)

        null_model_num_urb += [model_num_urb]

    return 1.0 * np.searchsorted(sorted(null_model_num_urb), num_urban) / len(null_model_num_urb)


def get_num_clusters(M):
    labels = measure.label(M, connectivity=1)
    regions = measure.regionprops(labels, cache=False)

    return len(regions)


def read_macro2ccdf(configs):
    fname = '{}/macro2ccdf_rescaled4.json'.format(configs['generated_files_path'])
    with open(fname, 'r') as f:
        macro2ccdf_rescaled4 = json.load(f)

    return macro2ccdf_rescaled4


def find_phase_class_from_tile(M, macro_array, orig_tile_km, size=1000):
    km_per_pixel = orig_tile_km / (size**2)
    urban_area_km2, num_urban, tile_km2 = M.sum() * km_per_pixel, get_num_clusters(M), orig_tile_km
    q = get_urban_phase_class(
        urban_area_km2, num_urban, tile_km2, macro_array, num_realisations=1000
    )
    return q


def phase2class(phase):
    cl = 4

    if 0.1 < phase <= 0.5:
        cl = 2
    elif 0.5 < phase < 0.9:
        cl = 3
    elif phase <= 0.1:
        cl = 1

    return cl


def retrieve_simulation(name, step, configs, temporal=False):
    d = '1000'
    if 'marco' in name:
        d = 'marco'
    M_sim1 = np.load(
        open('{}/simulations2steps/{}/{}'.format(configs['generated_files_path'], d, name), 'rb')
    )['M']
    if temporal:
        M_sim1[(M_sim1 <= step)] = 0
    else:
        M_sim1 = 1 * (M_sim1 > step)
    # M_sim1 = np.ma.array(M_sim1, mask=M_sim1 == 0)
    return M_sim1


def find_best_simulation_class(match, tileid, macro_array, orig_tile_km, model_type, configs):
    best_params = match[2][0]
    best_step = match[4][0]

    filename = create_name_file(1000, best_params[-1], best_params[0], best_params[1], model_type)
    M_sim = retrieve_simulation(filename, best_step, configs, temporal=False)
    q = find_phase_class_from_tile(M_sim, macro_array, orig_tile_km, size=M_sim.shape[0])

    return tileid, phase2class(q), q


def find_real_tile_class(tileid, macro_array, orig_tile_km, cache_dir):
    M = np.load(open('{}/{}.npz'.format(cache_dir, tileid), 'rb'))['M']

    # urban > 0
    if M.sum() == 0:
        return None, 0, 0

    q = find_phase_class_from_tile(M, macro_array, orig_tile_km, size=M.shape[0])

    return tileid, phase2class(q), q


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Compute the quantiles classes for all the simulations or real tiles"
    )

    parser.add_argument('--njobs', '-J', default=10, type=int)
    parser.add_argument('--model', '-M', default='1000', choices=['1000', 'rybski', 'marco'])
    parser.add_argument('--realtiles', dest='realtiles', action='store_true', help="Real tiles")
    parser.add_argument(
        '--no-realtiles', dest='realtiles', action='store_false', help="Simulation tiles"
    )
    parser.add_argument('--distance', '-D', default='earth', choices=['energy', 'earth'])

    parser.set_defaults(realtiles=False)
    return parser


def main():
    configs = load_config()

    parser = make_argument_parser()
    args = parser.parse_args()
    print("PARAMETERS", args)

    df_summary = pd.read_csv(
        '{}/summary_tiles_05x05.csv'.format(configs['generated_files_path']),
        dtype={'tileid': 'str'},
    )
    df_summary = df_summary.set_index('tileid')
    df_macro = pd.read_csv(
        '{}/macro_05x05.csv'.format(configs['generated_files_path']), dtype={'tileid': 'str'}
    )
    df_macro = df_macro.sort_values('tile_km2', ascending=False).groupby('tileid').first()
    df_macro = df_macro[['macro']]
    df_km2tiles = pd.read_csv(
        '{}/tiles_fullkm2_05x05.csv'.format(configs['generated_files_path']),
        dtype={'tileid': 'str'},
    )
    df_km2tiles = df_km2tiles.set_index('tileid')
    df_summary_macro = pd.merge(df_summary, df_macro, left_index=True, right_index=True)
    df_summary_macro = pd.merge(df_summary_macro, df_km2tiles, left_index=True, right_index=True)

    macro2ccdf_rescaled4 = read_macro2ccdf(configs)

    if args.realtiles:
        tiles = df_summary[df_summary['urban_area_km2'] > 0].index.values
        print("N tiles", len(tiles))

        matches = [
            [t, p, q]
            for t, p, q in Parallel(n_jobs=args.njobs, verbose=5)(
                delayed(find_real_tile_class)(
                    t,
                    macro2ccdf_rescaled4[df_summary_macro.loc[t, 'macro']],
                    df_summary_macro.loc[t, 'full_tile_km2'],
                    configs['tiles_cache_path'],
                )
                for t in tiles
            )
        ]

        output_filename = '{}/quantiles_classes.csv'.format(configs['generated_files_path'])

    else:
        json_comparisons = json.load(
            open(
                '{}/simulations/{}{}_1000x1000_lb0.01_hb1.0_all_1000.json'.format(
                    configs['generated_files_path'], args.distance, args.model
                )
            )
        )
        print("STEPS", len(json_comparisons))
        matches = [
            [t, p, q]
            for t, p, q in Parallel(n_jobs=args.njobs)(
                delayed(find_best_simulation_class)(
                    match,
                    tileid,
                    macro2ccdf_rescaled4[df_summary_macro.loc[tileid, 'macro']],
                    df_summary_macro.loc[tileid, 'full_tile_km2'],
                    args.model,
                    configs,
                )
                for tileid, match in tqdm(json_comparisons.items())
            )
        ]

        output_filename = '{}/quantiles_classes_{}_all_{}.csv'.format(
            configs['generated_files_path'], args.model, args.distance
        )

    # Write results
    with open(output_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['tileid', 'class', 'quantile'])
        for t, p, q in matches:
            if t:
                csvwriter.writerow([str(t), str(p), str(q)])


if __name__ == '__main__':
    main()
