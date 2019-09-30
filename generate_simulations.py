import cupy as cp
from joblib import Parallel, delayed
import itertools
import random
import os.path
import argparse
import time
import sys

ssw = sys.stdout.write
ssf = sys.stdout.flush
import numpy as np
from config_loader import *


class Rybski():

    def __init__(self, gamma1, gamma2, L, use_gpu=False, prob=0.5):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.L = L
        self.prob = prob
        self.twosteps = None
        self.is_second_step = False
        self.use_gpu = use_gpu

    def set_twosteps(self, val):
        self.twosteps = val

    def create_distance_matrix(self, L, maxdist=1e10, PBC=False):
        kL = L
        if kL > maxdist:
            kL = maxdist

        if PBC == True:
            dist = np.minimum(np.arange(kL), np.arange(kL, 0, -1))
        else:
            dist = np.arange(kL)
        dist *= dist
        dist_2d = np.sqrt(dist[:, None] + dist)

        if self.use_gpu:
            return cp.asarray(dist_2d)
        return dist_2d

    def simulate_rybski(self, M, stop_at_frac_compl, verbose=False):

        if self.twosteps is None:
            raise EnvironmentError("Set twosteps")
        elif self.twosteps and self.gamma2 is None:
            raise EnvironmentError("Set gamma2")

        dist = self.create_distance_matrix(self.L, maxdist=1e10, PBC=True).astype('float64')
        dist[0, 0] = 1.
        dist_gamma1 = dist ** -self.gamma1
        dist_gamma1[0, 0] = 0.

        dist_gamma2 = dist ** -self.gamma2
        dist_gamma2[0, 0] = 0.

        if verbose:
            start = time.time()
        max_steps = 1000
        cache_nonzeros = set()
        if self.use_gpu:
            cache_previous_step = cp.zeros_like(dist_gamma1)
        else:
            cache_previous_step = np.zeros_like(dist_gamma1)
        for t in range(max_steps):
            M1 = ((M > 0) * 1.).astype('float32')

            frac_complete = np.sum(M1) / self.L ** 2
            if frac_complete > stop_at_frac_compl:
                break
            if verbose:
                ssw(' %s -- %.3f - g %.3f s %s ...          \r' % (t, frac_complete, self.gamma2 if (
                            (self.twosteps and self.is_second_step) or (
                                not self.twosteps and np.random.uniform() >= self.prob)) else self.gamma1,
                                                                   stop_at_frac_compl))
                ssf()

            M0 = ((M < 0.5) * 1.)
            M += M1

            dist_gamma = dist_gamma1
            if (self.twosteps and self.is_second_step) or ((not self.twosteps) and np.random.uniform() >= self.prob):
                dist_gamma = dist_gamma2

            nonzeros = np.argwhere(M > 0)
            nonzeros = set(map(tuple, nonzeros))
            # Computes only the indexes not computed in the previous step
            if self.use_gpu:
                q = sum(
                    cp.roll(cp.roll(dist_gamma, i, axis=0), j, axis=1) for i, j in nonzeros.difference(cache_nonzeros))
            else:
                q = sum(np.roll(dist_gamma, (i, j), axis=(0, 1)) for i, j in nonzeros.difference(cache_nonzeros))
            # Sum from previous step
            q += cache_previous_step
            # Memorizes cache for the next step
            cache_nonzeros = set(nonzeros).union(cache_nonzeros)
            # Multiplies by the matrix
            cache_previous_step = q[:]
            if self.use_gpu:
                q = cp.asnumpy(q) * M0
            else:
                q = q * M0

            if stop_at_frac_compl < 0.01 and frac_complete < 0.001:  # and (not (self.twosteps and self.is_second_step))
                # 0.0005% of the area becomes urban at each step
                new_per_step = max(self.L ** 2 // (50000.), 1)
            elif frac_complete < 0.01:  # and (not (self.twosteps and self.is_second_step))
                # 0.005% of the area becomes urban at each step
                new_per_step = max(self.L ** 2 // (5000.), 1)
            else:
                # 1% of the area becomes urban at each step
                new_per_step = max(self.L ** 2 // (100.), 1)

            q /= np.sum(q)

            asd = np.random.multinomial(int(new_per_step), q.flatten())
            M += (np.reshape(asd, (self.L, self.L)) > 0.5) * 1.

        if verbose:
            ssw('\n');
            ssf()
            print(t, frac_complete, self.gamma1, self.gamma2)
            end = time.time()
            print("Elapsed time", end - start)

        return M


def create_name_file(L, S, gamma_1, gamma_2, twosteps, configs):
    dir = L
    filename = 'rybski_2steps'
    if not twosteps:
        dir = 'marco'
        filename = 'rybski_marco'

    return '{sim_dir}/{dir}/{filename}_{size}x{size}_s{S}_{gamma1}_{gamma2}.npz'.format(
        sim_dir=configs["simulations_path"], size=L, dir=dir, filename=filename, S=S, gamma1=gamma_1, gamma2=gamma_2)


def compute_simulation(gamma_1, gamma_2, S, max_urbanization, L, verbose, twosteps, use_gpu, gpuid, configs):
    if use_gpu:
        cp.cuda.Device(gpuid).use()

    filename = create_name_file(L, S, gamma_1, gamma_2, twosteps, configs)

    prob = 0.5
    stop_at_frac_compl = S
    if not twosteps:
        prob = S
        stop_at_frac_compl = max_urbanization

    ryb = Rybski(gamma1=gamma_1, gamma2=gamma_2, L=L, prob=prob, use_gpu=use_gpu)
    ryb.set_twosteps(twosteps)

    M0 = np.zeros((L, L), dtype='float32')
    M0[L // 2, L // 2] = 1

    if twosteps:
        # 1st stage
        M1 = ryb.simulate_rybski(M=M0, stop_at_frac_compl=stop_at_frac_compl, verbose=verbose)

        # 2nd stage
        ryb.is_second_step = True
        M = ryb.simulate_rybski(M=M1, stop_at_frac_compl=max_urbanization, verbose=verbose)
    else:
        M = ryb.simulate_rybski(M=M0, stop_at_frac_compl=stop_at_frac_compl, verbose=verbose)

    np.savez(filename, M=M.astype('float32'))

    return True


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Create simulations through the GPU or the CPU"
    )
    parser.add_argument('--njobs', '-J',
                        default=10, type=int)
    parser.add_argument('--size', '-S',
                        default=1000, type=int)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.add_argument('--twosteps', dest='twosteps', action='store_true', help="Two steps model")
    parser.add_argument('--no-twosteps', dest='twosteps', action='store_false', help="Probabilistic model")
    parser.add_argument('--gpuid', '-G',
                        default=0, type=int)
    parser.add_argument('--slist', nargs='+', type=float)

    parser.set_defaults(verbose=False, twosteps=True, gpu=False)
    return parser


def main():
    configs = load_config()

    parser = make_argument_parser()
    args = parser.parse_args()
    print("PARAMETERS", args)

    L = args.size
    if args.gpu:
        cp.cuda.Device(args.gpuid).use()

    # Sprawl: first compact, then random
    gamma_1 = [1., 1.4, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.6, 3.8, 4., 5., 6., 8., 10.]
    gamma_2 = gamma_1[:]
    S = [0.00005, 0.0001, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
         0.2, 0.3,
         0.4, 0.5]
    if args.slist:
        S = args.slist
    max_urbanization = 0.6

    if not args.twosteps:
        S = [0.5, 0.52, 0.54, 0.57, 0.61, 0.64, 0.66, 0.68, 0.71, 0.73, 0.75, 0.77, 0.79, 0.82, 0.84, 0.86, 0.89, 0.91,
             0.93, 0.96, 0.98]

    list_parameters = list(itertools.product(gamma_1, gamma_2, S))
    # Exclude special cases (repetitions)
    if not args.twosteps:
        list_parameters = [(g1, g2, s) for g1, g2, s in list_parameters if s != 0.5 or (s == 0.5 and g1 <= g2)]
    # Distribuite parameters
    random.shuffle(list_parameters)

    todo = [(g1, g2, s) for g1, g2, s in list_parameters if
            not os.path.isfile(create_name_file(L, s, g1, g2, args.twosteps, configs))]
    done = [(g1, g2, s) for g1, g2, s in list_parameters if
            os.path.isfile(create_name_file(L, s, g1, g2, args.twosteps, configs))]

    print("TODO:", len(todo), "DONE:", len(done))
    _ = [True for _ in Parallel(n_jobs=args.njobs, verbose=10, )(
        delayed(compute_simulation)(g1, g2, s, max_urbanization, L, args.verbose, args.twosteps, args.gpu, args.gpuid,
                                    configs) for g1, g2, s in todo)]


if __name__ == '__main__':
    main()
