#!/usr/bin/env bash

NJOBS=15
DIMENSION="05"
BASE_PATH="/home/denadai/ema"
VRT_PATH="/unreliable/nadai/datasets/ema/GUF+/myguf+.vrt"
OUTPUTRASTERDIR="/home/denadai/ema/data/GUF+/${DIMENSION}x${DIMENSION}"
TMP_RASTERS_DIR="/home/denadai/ema/tmp"
GDALCOMMANDS_DIR="/home/denadai/ema/data/generated_files/GDAL_commands"

preprocessing_OutputFile="translate_${DIMENSION}x${DIMENSION}.txt"
polygonize_OutputFile="polygonize_${DIMENSION}x${DIMENSION}.txt"

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
    echo -n > "${GDALCOMMANDS_DIR}/${polygonize_OutputFile}"
    find ${OUTPUTRASTERDIR}/ -type f -exec rm '{}' \;
    find ${TMP_RASTERS_DIR}/ -type f -exec rm '{}' \;
    #rm ${VRT_PATH}

    # Drop index
    echo -e "\nDROP INDEXES"
    psql ${CONNECTION_STRING} -c "DROP INDEX world075_gp_wkb_geometry_idx"
    psql ${CONNECTION_STRING} -c "DELETE FROM world075_gp"
    psql ${CONNECTION_STRING} -c "ALTER SEQUENCE world075_gp_ogc_fid_seq RESTART WITH 1;"

    gdalbuildvrt ${VRT_PATH} -srcnodata "0" -hidenodata -resolution highest /unreliable/nadai/datasets/ema/GUF+/v003/*.tif

    # Creates tif
    tiff_list=$(ogrinfo -ro -al "PG:${OGR_CONNECTION_STRING}" -sql "SELECT tileid::varchar from tiles_land where type='tile${DIMENSION}'" | grep 'tileid (' | sed -E 's/.*String[\)]\s=\s//g' | uniq );
    len=$(echo ${tiff_list} | wc -w)
    echo ${len}

    counter=0
    for id in ${tiff_list};
    do
       extent=$( psql ${CONNECTION_STRING} -t -c "SELECT ST_Extent(geom) from tiles_land where tileid =${id} AND type='tile${DIMENSION}'" | head -n 1 | sed 's/ BOX(//g' | tr -d ')' | sed 's/,/ /g' | awk -F ' ' '{print $1, $4, $3, $2}' );
       echo "gdal_translate -strict -a_nodata 0 -projwin $extent -of GTiff -q -co COMPRESS=PACKBITS ${VRT_PATH} ${TMP_RASTERS_DIR}/${id}.tif && python3 ${BASE_PATH}/gdal_calc.py -A ${TMP_RASTERS_DIR}/${id}.tif --quiet --overwrite --outfile=${OUTPUTRASTERDIR}/${id}.tif --calc=\"(A==1)\" --co=\"COMPRESS=PACKBITS\" --NoDataValue=255" >> "${GDALCOMMANDS_DIR}/${preprocessing_OutputFile}";
       echo "PG_USE_COPY=YES python3 ${BASE_PATH}/gdal_polygonize.py ${OUTPUTRASTERDIR}/${id}.tif -q -b 1 -f PostgreSQL PG:\"${OGR_CONNECTION_STRING}\" world075_gp && rm ${OUTPUTRASTERDIR}/${id}.tif && rm ${TMP_RASTERS_DIR}/${id}.tif" >> "${GDALCOMMANDS_DIR}/${polygonize_OutputFile}";
       ProgressBar ${counter} ${len}
       counter=$((counter + 1))
    done

    echo -e "\nPARALLEL TRANSLATE"
    nice parallel --bar --jobs ${NJOBS} < "${GDALCOMMANDS_DIR}/${preprocessing_OutputFile}"

    echo -e "\nPARALLEL POLYGONIZE"
    nice parallel --bar --jobs ${NJOBS} < "${GDALCOMMANDS_DIR}/${polygonize_OutputFile}"

    echo -e "\nCREATING INDEXES"
    # Create index
    psql ${CONNECTION_STRING} -c "CREATE INDEX ON world075_gp USING GIST (wkb_geometry)"
    psql ${CONNECTION_STRING} -c "CLUSTER world075_gp USING world075_gp_wkb_geometry_idx"

fi