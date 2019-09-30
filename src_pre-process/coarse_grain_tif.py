#!/usr/bin/env python
import os
from collections import Counter

import numpy as np
import networkx as nx
import json
from osgeo import osr, gdal
from joblib import Parallel, delayed
from sqlalchemy import *

homedir = os.path.expanduser('~/')


def tally(l):
	"""
	Tally occurrences of items in a list
	"""
	cnt = Counter()
	for i in l:
		cnt[i] += 1.
	return cnt


def neighbors(x, y):
	return ((x[0] - y[0]) ** 2. + (x[1] - y[1]) ** 2.) ** 0.5 < 1.2


def adjacency_list(M, minpop):
	populated = zip(*np.where(M > minpop))
	set_populated = set(populated)

	adj_list = []

	Li, Lj = np.shape(M)
	nn0 = np.array([[k1, k2] for k1 in range(-1, 2) for k2 in range(-1, 2) \
					if neighbors([0, 0], [k1, k2])], dtype=int)

	for (i, j) in set_populated:
		x = (i, j)
		for k1, k2 in nn0:
			if 0 <= i + k1 < Li and 0 <= j + k2 < Lj:
				y = (i + k1, j + k2)
				if y in set_populated:
					adj_list += [(x, y)]

	return adj_list


def get_clusters(M, minpop):
	adj_list = adjacency_list(M, minpop)

	G = nx.Graph()
	G.add_edges_from(adj_list)
	sorted_components = sorted(nx.connected_components(G), key=len, reverse=True)

	cluster_pop = [[M[p] for p in c] for c in sorted_components]

	clusters = [c for c in sorted_components]

	return clusters, cluster_pop


def mode(l):
	item2counts = tally(l)
	m = max(list(item2counts.values()))
	return np.random.choice([k for k, v in item2counts.items() if v == m])


def coarse_grain_matrix(M, kr, kc, func='average'):
	"""
	Input:
	------

	M        -->  np.array: 2D Numpy array
	kr       -->  float:    Number of Rows to merge
	kc       -->  float:    Number of Colums to merge
	func     -->  string:   function to apply to each block;
							if "average" gives the average, if "mode" gives the most common, else the sum (total)
	"""
	# np.repeat(numpy.add.reduceat(numpy.add.reduceat(M, numpy.arange(0, M.shape[0], kr), axis=0), numpy.arange(0, M.shape[1], kc), axis=1) / float(kr*kc),4,axis=0).repeat(4,axis=1)
	if func == 'average':
		return np.add.reduceat(np.add.reduceat(M, np.arange(0, M.shape[0], kr), axis=0), np.arange(0, M.shape[1], kc),
							   axis=1) / float(kr * kc)
	elif func == 'mode':
		h, w = M.shape
		b = M[:h - h % kr, :w - w % kc].reshape(h // kr, kr, -1, kc).swapaxes(1, 2).reshape(-1, kr * kc)
		return np.apply_along_axis(mode, 1, b).flatten().reshape(h // kr, w // kc)
	else:
		return np.add.reduceat(np.add.reduceat(M, np.arange(0, M.shape[0], kr), axis=0), np.arange(0, M.shape[1], kc),
							   axis=1)


def coarse_grain_matrix_mode3(M, kr, kc):
	U = coarse_grain_matrix(1. * (M == 1), kr, kc, func='sum')
	E = coarse_grain_matrix(1. * (M == 0), kr, kc, func='sum')
	W = coarse_grain_matrix(1. * (M == -1), kr, kc, func='sum')
	U += 1e-8 * np.random.random(size=U.shape)
	E += 1e-8 * np.random.random(size=E.shape)
	W += 1e-8 * np.random.random(size=W.shape)
	return 1. * ((U > E) & (U > W)) - 1. * ((W > E) & (W > U))


def renormalisation_steps(M, n_renorm, tile_type, kc=2, urban_px_val=1, empty_px_val=0, water_px_val=-1):
	"""
	M0  :  2D array of int (tile tiff)

	urban_empty_water  :  list of int, 1st element (urban) is the integer that corresponds to urban pixels in M,
		2nd element (empty) is the integer corresponding to land not yet urbanized, positive entries indicate urban area
		3rd element (water) is the integer corresponding to non-urbanizable land (water, slopes, ...),

	Return

	r2ta_tu_tc  :  dict, key is the renormalisation step
		values is the list [total urbanizable area (zeros+ones), total urban area (ones), total number of clusters]
	"""
	M0 = M.astype('float32')
	np.place(M0, M == urban_px_val, 1)
	np.place(M0, M == empty_px_val, 0)
	np.place(M0, M == water_px_val, -1)
	minpop = 0.5

	r2ta_tu_tc = {r: [] for r in range(n_renorm)}

	r2M = M0

	for r in range(1, n_renorm):
		new_r2M = coarse_grain_matrix_mode3(r2M, kc, kc)
		Mc = (new_r2M > 0).astype(int)

		if tile_type == '05' or r > 3:
			clusters, cluster_pop = get_clusters(Mc, minpop)
			r2ta_tu_tc[r] = [np.sum(1. * (new_r2M >= 0)), 1. * np.sum(Mc), 1. * len(cluster_pop)]
		r2M = new_r2M

	return r2ta_tu_tc


def renormalise_matrix(M, urban_empty_water, tileID, tile_type, urban_px_val, empty_px_val, water_px_val, outdir='./'):
	"""
	M  :  2D array of int (tile tiff)

	urban_empty_water  :  list of int, 1st element (urban) is the integer that corresponds to urban pixels in M,
		2nd element (empty) is the integer corresponding to land not yet urbanized, positive entries indicate urban area
		3rd element (water) is the integer corresponding to non-urbanizable land (water, slopes, ...),

	tileID  :  str, name of the tif file of tile M

	"""
	L = len(M[0])
	n_renorm = int(np.log2(1. * L / 10)) + 1  # to end with a 10x10 matrix at the last c-g step

	r2ta_tu_tc = renormalisation_steps(M, n_renorm, tile_type, urban_px_val=urban_px_val, empty_px_val=empty_px_val, water_px_val=water_px_val, kc=2)

	# save to file
	fname = outdir + '/r2ta_tu_tc_{}.json'.format(tileID)
	with open(fname, 'w') as f:
		json.dump(r2ta_tu_tc, f)


def open_tif_file(tiff):
	# Open tif file
	ds = gdal.Open(tiff)
	try:
		M = np.array(ds.GetRasterBand(1).ReadAsArray()).astype('int16')
	except AttributeError:
		print("Error while opening file %s." % tiff)
		return -1

	return M


def read_merge_geotiffs(tileID, urban_dir, water_dir, hydro_dir, steep_dir):
	# Read urban areas
	M = open_tif_file("{tif_dir}/{tileid}.tif".format(tif_dir=urban_dir, tileid=tileID))
	M_water = open_tif_file("{tif_dir}/{tileid}.tif".format(tif_dir=water_dir, tileid=tileID))
	M_steep = open_tif_file("{tif_dir}/{tileid}.tif".format(tif_dir=steep_dir, tileid=tileID))

	# Preprocess M
	np.place(M, M_water == 255, 1)
	np.place(M, M_steep == 1, 1)

	# Water bodies tif could not exist
	fname = "{tif_dir}/{tileid}.tif".format(tif_dir=hydro_dir, tileid=tileID)
	if os.path.isfile(fname):
		M_hydro = open_tif_file(fname)
		np.place(M, M_hydro == 255, 1)
	return M


def process_tile(tileID, urban_dir, water_dir, hydro_dir, steep_dir, outdir, tile_type):
	M = read_merge_geotiffs(tileID, urban_dir, water_dir, hydro_dir, steep_dir)
	renormalise_matrix(M, urban_px_val=0, empty_px_val=255, water_px_val=1, tileID=tileID, tile_type=tile_type, outdir=outdir)


def main():
	DIMENSION = '05'
	URBAN_DIR = '../data/GUF+/{d}x{d}'.format(d=DIMENSION)
	WATER_DIR = '../data/sea/{d}x{d}'.format(d=DIMENSION)
	HYDRO_DIR = '../data/hydro/{d}x{d}'.format(d=DIMENSION)
	STEEP_DIR = '../data/terrain_mask/{d}x{d}'.format(d=DIMENSION)
	OUTPUT_DIR = '../data/generated_files/coarse_grain/{d}x{d}'.format(d=DIMENSION)

	np.random.seed(42)

	engine = create_engine('postgresql://nadai@localhost:5432/ema')
	sql = text("select tileid from tiles_macros where (macro LIKE '% Europe') AND type='tile{}'".format(DIMENSION))
	result = engine.execute(sql)

	tiles = [r[0] for r in result]

	print(len(tiles), "DISPATCHED")

	Parallel(n_jobs=15, verbose=3)(
		delayed(process_tile)(tileid, URBAN_DIR, WATER_DIR, HYDRO_DIR, STEEP_DIR, OUTPUT_DIR, DIMENSION) for tileid in
		tiles)


if __name__ == '__main__':
	main()
