# Precise mapping, density and spatial structure of all human settlements on Earth

This repository contains all the code required to reproduce the results presented in the following paper:

* E. Strano, F. Simini, M. De Nadai, T. Esch, and M. Marconcini. *Precise mapping, density and spatial structure of all human settlements on Earth*, 2019.

## Getting Started

### Installation

- Clone this repository
```
git clone https://github.com/denadai2/precise-mapping-human-settlements
```

- Install the dependencies

``` sh
pip3 install -r requirements.txt
```

- If you want to have also the source dataset of all the urban areas install also:

* [PostgreSQL 10.0](https://www.postgresql.org/) 
* [PostGIS 2.4.1](https://postgis.net) extension


### Download intermediate datasets

- Download the [Multi parameter simulations](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CB1BW6) dataset and place it in `data/generated_files/simulations2steps/1000`:
- Download the [Multi-prob parameter simulations](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/S5XPHD) dataset and place it in `data/generated_files/simulations2steps/marco`:
- Download the [distances](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2KV3EG) dataset and place it in `data/generated_files/simulations/distances`:


# Code

The code of the analysis in divided in two parts: the Python scripts and modules used to support the analysis, and the notebooks where the outputs of the analysis have been produced.

## Scripts

* `generate_simulations.py` : script used to generate the simulated in parallel with the CPU or the GPUs.
* `generate_tiles_cache.py` : script used to create the cache for the real tiles and simulated tiles.
* `compute_quantiles.py` : script used to compute the quantile classes for all the tiles (real and simulated).
* `histogram_namedtuple.py` : support file.
* `histograms_compare_simulations_tiles.py` : script to find the matches between the real and simulated tiles.
* `notebooks/comparison_distances.ipynb` : script to compute all the distances and plot them.
* `notebooks/plot_distributions_macro.ipynb` : script to plot the global figures (Fig 2 and 3).
* `notebooks/simulation_figure.ipynb` : script to plot the simulations and real tiles.
* `notebooks/figures_precise_estimation.ipynb` : script to plot the simulations and real tiles.

## Extra scripts

* `src_pre-process/areas_filtering.md` : list of filterings done to the dataset.
* `src_pre-process/config.bash` : PostgreSQL connection configurations.
* `src_pre-process/incl.bash` : support file.
* `src_pre-process/polygonize_terrain_mask.bash` : file to polygonize (and sent to the DB) the terrain mask.
* `src_pre-process/polygonize_urban_areas.bash` : file to polygonize (and sent to the DB) the urban areas.
* `src_pre-process/rasterize_sea_tif.bash` : DB => TIF for sea areas.
* `src_pre-process/rasterize_sea_tif.bash` : DB => TIF for urban areas.
* `src_pre-process/prepare_urban_areas_list.py`: script to prepare the file `filippo_areas_reduced4.csv.gz`



## License
This code is licensed under the MIT license. 