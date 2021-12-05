#!/bin/bash
#
# Creates the tif from the database
#

NJOBS=25
DIMENSION="05"
OUTPUT_RASTERDIR1="/home/denadai/ema/deleteme/sea/${DIMENSION}x${DIMENSION}"
OUTPUT_RASTERDIR2="/home/denadai/ema/deleteme/hydro/${DIMENSION}x${DIMENSION}"
GDALCOMMANDS_DIR="data/generated_files/GDAL_commands"
preprocessing_OutputFile="rasterize_sea_${DIMENSION}x${DIMENSION}.txt"
hydro_OutputFile="rasterize_hydro_${DIMENSION}x${DIMENSION}.txt"

npixels=11134
if [ "${DIMENSION}" == "05" ]
then
    npixels=5567
fi

# Import utilities
DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/incl.bash"
. "$DIR/config.bash"

read -p "Are you sure to wipe out everything and start the script? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Clear files
    echo -n > "${GDALCOMMANDS_DIR}/${preprocessing_OutputFile}"
    echo -n > "${GDALCOMMANDS_DIR}/${hydro_OutputFile}"
    find ${OUTPUT_RASTERDIR1}/ -type f -exec rm '{}' \;
    find ${OUTPUT_RASTERDIR2}/ -type f -exec rm '{}' \;

    # Creates tif
    tiff_list=$(ogrinfo -ro -al "PG:${OGR_CONNECTION_STRING}" -sql "SELECT tileid::varchar from tiles_land where type='tile${DIMENSION}'" | grep 'tileid (' | sed -E 's/.*String[\)]\s=\s//g' | uniq );
    len=$(echo ${tiff_list} | wc -w)
    echo ${len}

    counter=0
    for id in ${tiff_list};
    do
       extent=$( psql ${CONNECTION_STRING} -t -c "SELECT ST_Extent(geom) from tiles_land where tileid =${id} AND type='tile${DIMENSION}'" | head -n 1 | sed 's/ BOX(//g' | tr -d ')' | sed 's/,/ /g' | awk -F ' ' '{print $1, $2, $3, $4}' );
       echo "gdal_rasterize -q -te ${extent} -ts ${npixels} ${npixels} -a_nodata 0 -burn 255 -i PG:${OGR_CONNECTION_STRING} -sql \"SELECT tile_real_geom FROM tiles_land WHERE tileid=${id} AND type='tile${DIMENSION}'\" -co COMPRESS=PACKBITS -ot Byte ${OUTPUT_RASTERDIR1}/${id}.tif" >> "${GDALCOMMANDS_DIR}/${preprocessing_OutputFile}";
       ProgressBar ${counter} ${len}
       counter=$((counter + 1))
    done

    echo "\n"
    # Creates tif
    tiff_list=$(ogrinfo -ro -al "PG:${OGR_CONNECTION_STRING}" -sql "SELECT tileid::varchar from tiles_coarse_water where type='tile${DIMENSION}'" | grep 'tileid (' | sed -E 's/.*String[\)]\s=\s//g' | uniq );
    len=$(echo ${tiff_list} | wc -w)
    echo ${len}

    counter=0
    for id in ${tiff_list};
    do
       extent=$( psql ${CONNECTION_STRING} -t -c "SELECT ST_Extent(geom) from tiles_land where tileid =${id} AND type='tile${DIMENSION}'" | head -n 1 | sed 's/ BOX(//g' | tr -d ')' | sed 's/,/ /g' | awk -F ' ' '{print $1, $2, $3, $4}' );
       echo "gdal_rasterize -q -te ${extent} -ts ${npixels} ${npixels} -a_nodata 0 -burn 255 PG:\"${OGR_CONNECTION_STRING}\" -sql \"SELECT geom FROM tiles_coarse_water WHERE tileid=${id} AND type='tile${DIMENSION}'\" -co COMPRESS=PACKBITS -ot Byte ${OUTPUT_RASTERDIR2}/${id}.tif" >> "${GDALCOMMANDS_DIR}/${hydro_OutputFile}";
       ProgressBar ${counter} ${len}
       counter=$((counter + 1))
    done

    echo -e "\nPARALLEL RASTERIZE"
    nice parallel --bar --jobs ${NJOBS} < "${GDALCOMMANDS_DIR}/${preprocessing_OutputFile}"
    nice parallel --bar --jobs ${NJOBS} < "${GDALCOMMANDS_DIR}/${hydro_OutputFile}"
fi
