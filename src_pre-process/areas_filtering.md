
```sql
CREATE MATERIALIZED VIEW urban_areas_extent AS
SELECT gid, tileid, ST_YMax(ST_Extent(i.geom)) as y_max, ST_YMin(ST_Extent(i.geom)) as y_min, ST_XMax(ST_Extent(i.geom)) as x_max, ST_XMin(ST_Extent(i.geom)) as x_min, abs(ST_XMax(i.geom)-ST_XMin(i.geom)) <= 0.0005 as condX
FROM urban_areas_view i 
WHERE abs(ST_YMax(i.geom)-ST_YMin(i.geom)) <= 0.0005 OR abs(ST_XMax(i.geom)-ST_XMin(i.geom)) <= 0.0005
AND i.type='tile05'
GROUP BY tileid, gid, geom;
CREATE INDEX ON urban_areas_extent (gid DESC);
CREATE INDEX ON urban_areas_extent (condx);
CREATE INDEX ON urban_areas_extent (y_max, y_min);
CREATE INDEX ON urban_areas_extent (x_max, x_min);
CREATE INDEX ON urban_areas_extent (tileid);


CREATE MATERIALIZED VIEW urban_areas_cleaner AS
SELECT tileid, gid, ST_XMax(geom) as x_max, ST_XMin(geom) AS x_min, ST_YMax(geom) as y_max, ST_YMin(geom) as y_min
FROM urban_areas_view
WHERE type='tile05';
CREATE INDEX ON urban_areas_cleaner (y_max, y_min);
CREATE INDEX ON urban_areas_cleaner (x_max, x_min);
CREATE INDEX ON urban_areas_cleaner (tileid);
CREATE INDEX ON urban_areas_cleaner (gid);

create materialized view todelete1 AS
SELECT a.tileid, COUNT(*) as c, array_agg(a.gid) as gids, a.x_max, a.x_min
FROM urban_areas_extent a 
WHERE a.condx
GROUP BY a.tileid, a.x_max, a.x_min
HAVING COUNT(*) > 15;
CREATE INDEX ON todelete1 (x_max, x_min);
CREATE INDEX ON todelete1 (tileid);

create materialized view todelete2 AS
SELECT a.tileid, COUNT(*) as c, array_agg(a.gid) as gids, a.y_max, a.y_min
FROM urban_areas_extent a 
WHERE NOT a.condx
GROUP BY a.tileid, a.y_max, a.y_min
HAVING COUNT(*) > 15;
CREATE INDEX ON todelete2 (y_max, y_min);
CREATE INDEX ON todelete2 (tileid);


create materialized view todelete3 AS
select o.*
FROM todelete1 o
WHERE NOT EXISTS(SELECT 1 FROM urban_areas_cleaner n WHERE n.tileid=o.tileid AND (n.x_max > o.x_max AND n.x_min <= o.x_min));

create materialized view todelete4 AS
select o.*
FROM todelete2 o
WHERE NOT EXISTS(SELECT 1 FROM urban_areas_cleaner n WHERE n.tileid=o.tileid AND ((n.y_max >= o.y_max AND n.y_min < o.y_min) OR (n.y_max > o.y_max AND n.y_min <= o.y_min)));


DELETE FROM urban_areas USING todelete3 d WHERE gid = ANY(d.gids);
DELETE FROM urban_areas USING todelete4 d WHERE gid = ANY(d.gids);
DELETE FROM urban_areas_tiles USING todelete3 d WHERE gid = ANY(d.gids);
DELETE FROM urban_areas_tiles USING todelete4 d WHERE gid = ANY(d.gids);

-------


CREATE MATERIALIZED VIEW delete1 AS 
SELECT a.gid, COUNT(*) as c, array_agg(b.gid) as gids
FROM urban_areas_extent a
INNER JOIN urban_areas_extent b ON b.x_max = a.x_max AND b.x_min = a.x_min AND a.tileid = b.tileid AND a.gid < b.gid AND a.condX = b.condX
WHERE NOT EXISTS(SELECT 1 FROM urban_areas_cleaner n WHERE n.tileid=a.tileid AND ((n.x_max >= a.x_max AND n.x_min < a.x_min) OR (n.x_max > a.x_max AND n.x_min <= a.x_min))) AND (a.condX) 
GROUP BY a.gid
HAVING COUNT(*) > 15;


CREATE MATERIALIZED VIEW delete2 AS 
SELECT a.gid, COUNT(*) as c, array_agg(b.gid) as gids
FROM urban_areas_extent a
INNER JOIN urban_areas_extent b ON b.y_max = a.y_max AND b.y_min = a.y_min AND a.tileid = b.tileid AND a.gid < b.gid AND a.condX = b.condX
WHERE NOT EXISTS(SELECT 1 FROM urban_areas_cleaner n WHERE n.tileid=a.tileid AND ((n.y_max >= a.y_max AND n.y_min < a.y_min) OR (n.y_max > a.y_max AND n.y_min <= a.y_min))) AND (NOT a.condX)
GROUP BY a.gid
HAVING COUNT(*) > 15;

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



