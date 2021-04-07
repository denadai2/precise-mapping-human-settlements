import numpy as np
from joblib import Parallel, delayed
import os

from os import listdir
from os.path import isfile, join
import argparse
from skimage import measure
import gdal
from config_loader import *
from PIL import Image
from tqdm import tqdm


def open_tif_file(tiff):
    # Open tif file
    ds = gdal.Open(tiff)
    try:
        M = np.array(ds.GetRasterBand(1).ReadAsArray()).astype('int16')
    except AttributeError:
        print("Error while opening file %s." % tiff)
        return -1

    return M


def get_clusters_area(M):
    labels = measure.label(M, connectivity=1)
    regions = measure.regionprops(labels, cache=False)
    cluster_pop = np.zeros(len(regions), dtype='int32')
    for i, x in enumerate(regions):
        cluster_pop[i] = x.area

    return cluster_pop


def compute_cache_simulation_tiles(input_path, output_path, filename):

    save_filename = '{}/{}'.format(output_path, filename)
    if os.path.isfile(save_filename):
        return True

    fload = np.load('{}/{}'.format(input_path, filename))
    M = fload['M'].astype('float32')
    L = M.shape[0]
    nsteps = int(np.max(M) - 2)

    # Percentage urbanization
    perc_array = np.zeros(nsteps, 'float32')
    urb_steps = {}
    for s in range(nsteps):
        Mtemp = (1. * (M > s))
        perc_array[s] = np.sum(Mtemp)/(L**2)
        urb_steps['s{}'.format(s)] = get_clusters_area(Mtemp)

    np.savez_compressed(save_filename, perc=perc_array, M=M, **urb_steps)

    return True


def compute_cache_real_tiles(input_path, output_path, filename):

    save_filename = '{}/{}'.format(output_path, filename.replace('tif', 'npz'))
    if os.path.isfile(save_filename):
        return True

    M = open_tif_file('{}/{}'.format(input_path, filename))
    M = ((M == 0) * 1).astype('uint8')
    im = Image.fromarray(M*255, mode='L').convert('1')
    M = np.asarray(im.resize((1000, 1000), Image.NEAREST), dtype='uint8')

    L = M.shape[0]
    assert L == 1000
    perc_urb = np.sum(M)/(L**2)
    assert 0 <= perc_urb <= 1

    np.savez_compressed(save_filename, purb=perc_urb, M=M)

    return True


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Launch cache generation for simulated and real tiles"
    )
    parser.add_argument('--njobs', '-J',
                        default=10, type=int)
    parser.add_argument('--size', '-S',
                        default=1000, type=int)
    parser.add_argument('--twosteps', dest='twosteps', action='store_true', help="Two steps model")
    parser.add_argument('--no-twosteps', dest='twosteps', action='store_false', help="Probabilistic model")
    parser.add_argument('--realtiles', dest='realtiles', action='store_true', help="Real tiles cache generation")
    parser.add_argument('--no-realtiles', dest='realtiles', action='store_false', help="Simulation tiles cache generation")

    parser.set_defaults(twosteps=True, realtiles=False)
    return parser


def main():
    configs = load_config()

    parser = make_argument_parser()
    args = parser.parse_args()
    print("PARAMETERS", args)

    model_var = args.size
    if not args.twosteps:
        model_var = 'marco'

    if args.realtiles:
        input_path = configs['rasterized_tiles_path']
        output_path = configs['tiles_cache_path']
        onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f)) and 'tif' in f]
        print("N Real tiles", len(onlyfiles))

        _ = [True for _ in Parallel(n_jobs=args.njobs)(
            delayed(compute_cache_real_tiles)(input_path, output_path, f) for f in tqdm(onlyfiles))]

    input_path = '{}/{}'.format(configs["simulations_path"], model_var)
    output_path = '{}/{}'.format(configs["simulations_cache_path"], model_var)
    onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f)) and 'npz' in f]
    print("N Simulations", len(onlyfiles))

    # Filter for those files that are not existing
    onlyfiles = [f for f in onlyfiles if not os.path.isfile('{}/{}'.format(output_path, f.replace('tif', 'npz')))]

    _ = [True for _ in Parallel(n_jobs=args.njobs)(delayed(compute_cache_simulation_tiles)(input_path, output_path, f) for f in tqdm(onlyfiles))]


if __name__ == '__main__':
    main()
