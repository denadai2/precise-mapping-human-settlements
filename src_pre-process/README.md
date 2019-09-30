# Step by step
Edit `config.bash` and then the configuration strings of each bash file you will execute.

## 1- Import rasters into PostGIS

```bash
bash polygonize_urban_areas.sh
```

```sql
DROP index urban_areas_geom_idx;

INSERT INTO urban_areas (gid, geom, urban_area_km2)
SELECT ogc_fid as gid, wkb_geometry as geom, (ST_AREA(wkb_geometry::geography)/1000000.0)::NUMERIC(13,6)
FROM world075_gp w;

CREATE INDEX ON urban_areas USING GIST (geom);
```

Clean the DB

```sql
DELETE FROM urban_areas
WHERE (abs(ST_YMax(geom)-ST_YMin(geom)) >= 0.0006 AND abs(ST_XMax(geom)-ST_XMin(geom)) <= 9.08304157104235e-05)
OR (abs(ST_XMax(geom)-ST_XMin(geom)) >= 0.0006 AND abs(ST_YMax(geom)-ST_YMin(geom)) <= 9.08304157104235e-05);
```

```sql
DELETE FROM urban_areas
WHERE gid IN (SELECT gid                                                      
FROM (SELECT gid, geom, ST_Extent(geom) as extent from urban_areas GROUP BY gid, geom) as tmp
WHERE 
(abs(ST_XMax(extent)-ST_XMin(extent)) > 0.45 AND abs(ST_YMax(extent)-ST_YMin(extent)) < 0.1)
OR 
(abs(ST_YMax(extent)-ST_YMin(extent)) > 0.45 AND abs(ST_XMax(extent)-ST_XMin(extent)) < 0.1));
```

```sql
create materialized view oversize20192 as 
SELECT a.gid, COUNT(*) as c
FROM (SELECT geom, gid, tileid, ST_XMax(ST_Extent(i.geom)) as xmax, ST_XMin(ST_Extent(i.geom)) as xmin FROM urban_areas_view i WHERE abs(ST_XMax(i.geom)-ST_XMin(i.geom)) <= 0.0005 GROUP BY tileid, gid, geom) a
INNER JOIN (select gid, tileid, ST_XMax(ST_Extent(u.geom)) as xmax, ST_XMin(ST_Extent(u.geom)) as xmin 
FROM urban_areas_view u GROUP BY tileid, gid, geom) b ON b.xmax = a.xmax AND b.xmin = a.xmin AND a.gid != b.gid AND a.tileid = b.tileid
GROUP BY a.gid
HAVING COUNT(*) > 10;
```

```sql
create materialized view oversize20193 as 
SELECT a.gid, COUNT(*) as c
FROM (SELECT geom, gid, tileid, ST_YMax(ST_Extent(i.geom)) as ymax, ST_YMin(ST_Extent(i.geom)) as ymin FROM urban_areas_view i WHERE abs(ST_YMax(i.geom)-ST_YMin(i.geom)) <= 0.0005 GROUP BY tileid, gid, geom) a
INNER JOIN (select gid, tileid, ST_YMax(ST_Extent(u.geom)) as ymax, ST_YMin(ST_Extent(u.geom)) as ymin 
FROM urban_areas_view u GROUP BY tileid, gid, geom) b ON b.ymax = a.ymax AND b.ymin = a.ymin AND a.gid != b.gid AND a.tileid = b.tileid
GROUP BY a.gid
HAVING COUNT(*) > 10;
```

```sql
DELETE FROM urban_areas WHERE gid IN (SELECT gid FROM oversize20192 WHERE c > 30);
DELETE FROM urban_areas WHERE gid IN (SELECT gid FROM oversize20193 WHERE c > 30);
```

Connect the tiles with the urban areas
```sql
INSERT INTO urban_areas_tiles (gid, tileid, type) 
SELECT gid, tileid, 'tile05'
FROM (
    SELECT gid, t.tileid
    FROM urban_areas w
    INNER JOIN tiles_land t ON ST_CONTAINS(t.tile_real_geom, w.geom) 
    WHERE t.type = 'tile05'
    
    UNION ALL

    select gid, tileid from(
        SELECT
          gid, geom, tileid, ROW_NUMBER() OVER (PARTITION BY gid ORDER BY area DESC) AS r
        from (
            select c.gid, c.geom as geom, a.tileid, ST_Area(ST_Intersection(c.geom, a.geom)) as area
            from urban_areas as c 
            inner join tiles_land as a on ST_Intersects(a.tile_real_geom, c.geom) AND NOT ST_CONTAINS(a.tile_real_geom, c.geom)
            WHERE a.type = 'tile05'
        ) as dtable
        order by area
    ) x
    WHERE x.r = 1
) outside_query;
```

```sql
create materialized view oversize2019 as
select geom, tileid, perc, c
FROM (

select t.geom, t.tileid, (ws.c/tot.c::float)::float as perc, ws.c, RANK() OVER (PARTITION BY ws.tileid ORDER BY ws.c DESC) c4rank
from tiles_land t, 
(select tileid, abs(ST_XMax(geom)-ST_XMin(geom)) as w, count(*) as c
from
urban_areas_view
where type = 'tile05'
and abs(ST_XMax(geom)-ST_XMin(geom)) < 0.0001 
GROUP BY tileid, w
ORDER BY c DESC) as ws,
(select tileid, count(*) as c
from
urban_areas_view
where type = 'tile05' 
GROUP BY tileid 
HAVING sum(urban_area_km2) > 0.001) tot
where tot.tileid = ws.tileid and t.tileid = ws.tileid and tot.c > 0 and 
t.type = 'tile05' ) ext
where c4rank = 1
;
```

```sql
DELETE FROM urban_areas USING urban_areas_tiles u WHERE u.gid = urban_areas.gid AND u.tileid IN (select tileid FROM oversize2019 WHERE perc > 0.75 AND c > 10);
DELETE FROM urban_areas_tiles a WHERE NOT EXISTS (select 1 from urban_areas b where b.gid = a.gid);
```


## 2- Going global


## 3- Create summaries

### 3.1 - Macroareas
```sql
CREATE MATERIALIZED VIEW macro_urban_areas_05x05 AS
SELECT macro, tileid, SUM(urban_area_km2::NUMERIC(13,6)) as urban_area_km2, SUM(urban_areas_num) as urban_areas_num
FROM (
    SELECT u.tileid, t.macro, SUM(COALESCE(urban_area_km2, 0)::NUMERIC(13,6)) as urban_area_km2, COUNT(*) as urban_areas_num
    FROM urban_areas_view u
    INNER JOIN tiles_in_macro t ON u.tileid = ANY(t.tileids) AND u.type = t.type
    WHERE t.has_one_macro = 1 AND t.type = 'tile05'
    GROUP BY u.tileid, macro

    UNION ALL

    SELECT u.tileid, t.macro, SUM(COALESCE(urban_area_km2, 0)::NUMERIC(13,6)) as urban_area_km2, COUNT(*) as urban_areas_num
    FROM urban_areas_view u
    INNER JOIN tiles_macros_geom t ON u.tileid = t.tileid AND ST_CoveredBy(u.geom, t.tile_macro_geom) AND u.type = t.type
    WHERE t.type = 'tile05' AND t.has_one_macro = 0
    GROUP BY u.tileid, macro
) temp
GROUP BY tileid, macro;
CREATE INDEX ON macro_urban_areas_05x05 (tileid);
CREATE INDEX ON macro_urban_areas_05x05 (macro);
```

```sql
CREATE MATERIALIZED VIEW macro_urban_areas_1x1 AS
SELECT macro, tileid, SUM(urban_area_km2::NUMERIC(13,6)) as urban_area_km2, SUM(urban_areas_num) as urban_areas_num
FROM (
    SELECT u.tileid, t.macro, SUM(COALESCE(urban_area_km2, 0)::NUMERIC(13,6)) as urban_area_km2, COUNT(*) as urban_areas_num
    FROM urban_areas_view u
    INNER JOIN tiles_in_macro t ON u.tileid = ANY(t.tileids) AND u.type = t.type
    WHERE t.has_one_macro = 1 AND t.type = 'tile1'
    GROUP BY u.tileid, macro

    UNION ALL

    SELECT u.tileid, t.macro, SUM(COALESCE(urban_area_km2, 0)::NUMERIC(13,6)) as urban_area_km2, COUNT(*) as urban_areas_num
    FROM urban_areas_view u
    INNER JOIN tiles_macros_geom t ON u.tileid = t.tileid AND ST_CoveredBy(u.geom, t.tile_macro_geom) AND u.type = t.type
    WHERE t.type = 'tile1' AND t.has_one_macro = 0
    GROUP BY u.tileid, macro
) temp
GROUP BY tileid, macro;
CREATE INDEX ON macro_urban_areas_1x1 (tileid);
CREATE INDEX ON macro_urban_areas_1x1 (macro);
```


```sql
create materialized view macro_05x05 AS
SELECT macro, tileid, SUM(urban_area_km2::NUMERIC(13,6)) as urban_area_km2, SUM(num_urban) as num_urban, SUM(tile_km2::NUMERIC(13,6)) as tile_km2, tile_macro_geom
FROM (
    SELECT t.macro, t.tileid, COALESCE(u.urban_area_km2, 0) as urban_area_km2, COALESCE(u.urban_areas_num, 0) as num_urban, (t.tile_km2 - COALESCE(c.noland_km2, 0) - COALESCE(s.steep_km2, 0))::NUMERIC(13,6) as tile_km2, t.tile_macro_geom
    FROM tiles_macros_geom t
    LEFT JOIN macro_urban_areas_05x05 u ON t.macro = u.macro AND t.tileid = u.tileid
    LEFT JOIN macro_tiles_steep s ON t.macro = s.macro AND t.tileid = s.tileid
    LEFT JOIN macro_tiles_cannoturban c ON t.macro = c.macro AND t.tileid = c.tileid
    WHERE t.type = 'tile05'
) temp
GROUP BY macro, tileid, tile_macro_geom;
```

```sql
copy (select macro, tileid, urban_area_km2, num_urban, tile_km2 from macro_05x05) TO '/tmp/macro_05x05.csv' DELIMITER ',' CSV HEADER;
```

```bash
cp /tmp/macro_05x05.csv data/generated_files/macro_05x05.csv
```

### 3.2 - Tiles
```sql
create materialized view summary_tiles_05x05 AS
SELECT i.tileid, i.urban_area_km2, i.urban_areas_num, (i.tile_km2 - COALESCE(nl.noland_km2, 0) - COALESCE(s.steep_km2, 0)::NUMERIC(13,6)) as tile_km2, original_km2
FROM (
    SELECT t.tileid, t.type, SUM(COALESCE(c.urban_area_km2, 0)::NUMERIC(13,6)) as urban_area_km2, COUNT(c.*) as urban_areas_num, t.tile_km2, (ST_AREA(t.geom::geography)/1000000)::NUMERIC(13,6) as original_km2
    FROM tiles_land t
    LEFT JOIN urban_areas_view c on t.tileid = c.tileid AND c.type=t.type
    WHERE t.type = 'tile05'
    GROUP BY t.tileid, t.type, t.tile_km2, original_km2
) i                                            
LEFT JOIN tiles_steep s ON i.tileid = s.tileid AND i.type = s.type
LEFT JOIN tiles_cannoturban nl ON i.tileid = nl.tileid AND i.type = nl.type;
```

```sql
create materialized view summary_tiles_1x1 AS
SELECT i.tileid, i.urban_area_km2, i.urban_areas_num, (i.tile_km2 - COALESCE(nl.noland_km2, 0) - COALESCE(s.steep_km2, 0)::NUMERIC(13,6)) as tile_km2, i.tile_km2 as original_km2
FROM (
    SELECT t.tileid, t.type, SUM(COALESCE(c.urban_area_km2, 0)::NUMERIC(13,6)) as urban_area_km2, COUNT(c.*) as urban_areas_num, t.tile_km2
    FROM tiles_land t
    LEFT JOIN urban_areas_view c on t.tileid = c.tileid AND c.type=t.type
    WHERE t.type = 'tile1'
    GROUP BY t.tileid, t.type, t.tile_km2
) i                                            
LEFT JOIN tiles_steep s ON i.tileid = s.tileid AND i.type = s.type
LEFT JOIN tiles_cannoturban nl ON i.tileid = nl.tileid AND i.type = nl.type;
```

```bash
rm data/generated_files/summary_tiles_05x05.*
ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:4326 data/generated_files/summary_tiles_05x05.shp PG:"dbname='ema' user='denadai' port='50013' password='lollone'" -sql "select s.tileid, s.urban_area_km2, s.urban_areas_num as urban_num, s.tile_km2, t.geom, (case when s.tile_km2 <= 0 then 0 else s.urban_area_km2/s.tile_km2::NUMERIC(13,6) end) as perc_urban from summary_tiles_05x05 s inner join tiles_land t on t.tileid=s.tileid and t.type='tile05'"
```
```bash
rm /data/nadai/ema/data/generated_files/summary_tiles_1x1.*
ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:4326 /data/nadai/ema/data/generated_files/summary_tiles_1x1.shp PG:"dbname='ema' user='nadai'" -sql "select s.tileid, s.urban_area_km2, s.urban_areas_num as urban_num, s.tile_km2, t.geom, s.urban_area_km2/s.tile_km2::NUMERIC(13,6) as perc_urban from summary_tiles_1x1 s inner join tiles_land t on t.tileid=s.tileid and t.type='tile1'"
```

```sql
copy (select tileid, (case when tile_km2 <= 0 then 0 else urban_area_km2 end) as urban_area_km2, (case when tile_km2 <= 0 then 0 else urban_areas_num end) as urban_areas_num, (case when tile_km2 <= 0 then 0 else tile_km2 end) as tile_km2, original_km2 from summary_tiles_05x05) TO '/tmp/summary_tiles_05x05.csv' DELIMITER ',' CSV HEADER; 
```

```bash
cp /tmp/summary_tiles_05x05.csv data/generated_files/summary_tiles_05x05.csv
```

```sql
copy (select tileid, (ST_AREA(t.geom::geography)/1000000.0) as full_tile_km2 from tiles_land t where type='tile05') TO '/tmp/tiles_fullkm2_05x05.csv' DELIMITER ',' CSV HEADER; 
```

```bash
cp /tmp/tiles_fullkm2_05x05.csv data/generated_files/tiles_fullkm2_05x05.csv
```

```sql
copy (select tileid, urban_area_km2 from urban_areas_view where type='tile05') TO '/tmp/tiles_fullkm2_05x05.csv' DELIMITER ',' CSV HEADER; 
```

## 4- Stats
Urban percentage: 1.474983848026523474688154
```sql
select avg(case when tile_km2 <= 0 then 0 when urban_area_km2 > tile_km2 then 100 else (urban_area_km2/tile_km2 end) from summary_tiles_05x05;
```

Urban area: 1107682.428039
```sql
select sum(case when tile_km2 <= 0 then 0 else urban_area_km2 end) from summary_tiles_05x05;
```

Dry land: 131331424.371022
```sql
select sum(tile_km2) from tiles_land where type='tile05';
```

Dry land - steep - water: 106445525.043325
```sql
select sum(case when tile_km2 <= 0 then 0 else tile_km2 end) from summary_tiles_05x05;
```

### Shapefiles

```sh
ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:4326 data/generated_files/shps/macro_05x05.shp PG:"dbname='ema' user='nadai'" -sql "select s.macro, s.tileid, s.urban_area_km2, s.num_urban, s.tile_km2, tile_macro_geom as geom, CASE s.tile_km2
   WHEN 0 THEN 0
   ELSE s.urban_area_km2/s.tile_km2::NUMERIC(13,6)
END as perc_urban from macro_05x05 s inner join tiles_land t on t.tileid=s.tileid and t.type='tile05'"
```

## List of km2 per tiles
```sql
copy (select tileid, (ST_AREA(t.geom::geography)/1000000.0) as full_tile_km2 from tiles_land t where type='tile05') TO '/tmp/tiles_fullkm2_05x05.csv' DELIMITER ',' CSV HEADER;
```

```sh
cp /tmp/tiles_fullkm2_05x05.csv data/generated_files/tiles_fullkm2_05x05.csv
```

## Prepare other files

```sh
python3 prepare_urban_areas_list.py
```


## Extract shapefiles for the maps

```sql
CREATE MATERIALIZED VIEW tiles_land_without_water AS
SELECT i.tileid, i.type, CASE WHEN i.water_geom IS NULL THEN i.tile_real_geom ELSE ST_Difference(i.tile_real_geom, i.water_geom) END as tile_real_geom
FROM (
    SELECT t.tileid, t.type, ST_Union(w.geom) as water_geom, t.tile_real_geom 
    FROM tiles_land t
    LEFT JOIN water_etsimila w ON ST_Intersects(w.geom, t.tile_real_geom) AND NOT ST_Touches(w.geom, t.tile_real_geom)
    GROUP BY t.tileid, t.type, t.tile_real_geom
) i
WHERE i.type='tile05' OR i.type='tile1';
```


```sh
copy (select tileid, (ST_AREA(geom::geography)/1000000)::NUMERIC(13,6) as total_tile_area from tiles_land WHERE type='tile05') TO 'data/generated_files/filippo_totarea_05x05.csv' DELIMITER ',' CSV HEADER;
```


