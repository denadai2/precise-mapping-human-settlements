#!/usr/bin/env bash
#
# Creates the tif from the database
#

NJOBS=15
DIMENSION="05"
OUTPUTRASTERDIR="/home/denadai/ema/data/GUF+/${DIMENSION}x${DIMENSION}_from_postgis"
GDALCOMMANDS_DIR="/home/denadai/ema/data/generated_files/GDAL_commands"

preprocessing_OutputFile="postgist_rasterize_${DIMENSION}x${DIMENSION}.txt"

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
    find ${OUTPUTRASTERDIR}/ -type f -exec rm '{}' \;

    # Creates tif
    tiff_list=$(psql ${CONNECTION_STRING} -t -A -c "SELECT tileid::varchar from urban_areas_view where type='tile${DIMENSION}' GROUP BY tileid HAVING count(*) > 0")
    len=$(echo ${tiff_list} | wc -w)
    echo ${len}

    counter=0
    for id in ${tiff_list};
    do
       extent=$( psql ${CONNECTION_STRING} -t -c "SELECT ST_Extent(geom) from tiles_land where tileid =${id} AND type='tile${DIMENSION}'" | head -n 1 | sed 's/ BOX(//g' | tr -d ')' | sed 's/,/ /g' | awk -F ' ' '{print $1, $2, $3, $4}' );
       echo "gdal_rasterize -q -te ${extent} -ts 5567 5567 -a_nodata 0 -burn 255 -i PG:\"${OGR_CONNECTION_STRING}\" -sql \"SELECT geom FROM urban_areas_view WHERE tileid=${id} AND type='tile${DIMENSION}'\" -co COMPRESS=PACKBITS -ot Byte ${OUTPUTRASTERDIR}/${id}.tif" >> "${GDALCOMMANDS_DIR}/${preprocessing_OutputFile}";
       ProgressBar ${counter} ${len}
       counter=$((counter + 1))
    done

    echo -e "\nPARALLEL TRANSLATE"
    nice parallel --bar --jobs ${NJOBS} < "${GDALCOMMANDS_DIR}/${preprocessing_OutputFile}"
fi