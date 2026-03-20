-- === 1) CLEARING DATA ===
-- Date format MM/DD/YYYY to YYYY-MM-DD
-- Conversion of Price to a Real number

-- Telefonica
DROP VIEW IF EXISTS v_telefonica_clean;
CREATE VIEW v_telefonica_clean AS
SELECT
  substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
  CAST(REPLACE(Price, ',', '') AS REAL) AS p
FROM telefonica
WHERE Price IS NOT NULL
ORDER BY d;

-- Repsol
DROP VIEW IF EXISTS v_repsol_clean;
CREATE VIEW v_repsol_clean AS
SELECT
  substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
  CAST(REPLACE(Price, ',', '') AS REAL) AS p
FROM repsol
WHERE Price IS NOT NULL
ORDER BY d;

-- Inditex
DROP VIEW IF EXISTS v_inditex_clean;
CREATE VIEW v_inditex_clean AS
SELECT
  substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
  CAST(REPLACE(Price, ',', '') AS REAL) AS p
FROM inditex
WHERE Price IS NOT NULL
ORDER BY d;

-- Iberdrola
DROP VIEW IF EXISTS v_iberdrola_clean;
CREATE VIEW v_iberdrola_clean AS
SELECT
  substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
  CAST(REPLACE(Price, ',', '') AS REAL) AS p
FROM iberdrola
WHERE Price IS NOT NULL
ORDER BY d;

-- ACS
DROP VIEW IF EXISTS v_acs_clean;
CREATE VIEW v_acs_clean AS
SELECT
  substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
  CAST(REPLACE(Price, ',', '') AS REAL) AS p
FROM acs
WHERE Price IS NOT NULL
ORDER BY d;

-- BBVA
DROP VIEW IF EXISTS v_bbva_clean;
CREATE VIEW v_bbva_clean AS
SELECT
  substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
  CAST(REPLACE(Price, ',', '') AS REAL) AS p
FROM bbva
WHERE Price IS NOT NULL
ORDER BY d;

-- Amadeus
DROP VIEW IF EXISTS v_amadeus_clean;
CREATE VIEW v_amadeus_clean AS
SELECT
  substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
  CAST(REPLACE(Price, ',', '') AS REAL) AS p
FROM amadeus
WHERE Price IS NOT NULL
ORDER BY d;

-- Gold
DROP VIEW IF EXISTS v_gold_clean;
CREATE VIEW v_gold_clean AS
SELECT
  substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
  CAST(REPLACE(Price, ',', '') AS REAL) AS p
FROM gold
WHERE Price IS NOT NULL
ORDER BY d;

-- === 2) BONDS (SPAIN 10Y) CLEANING ===
-- Compute adjusted real price (not yields)
-- True price = -8 * ((B[i] - B[i-1]) / 100.0)

DROP VIEW IF EXISTS v_bonds_clean;
CREATE VIEW v_bonds_clean AS
WITH raw AS (
  SELECT
    substr(Date,7,4) || '-' || substr(Date,1,2) || '-' || substr(Date,4,2) AS d,
    CAST(REPLACE(Price, ',', '') AS REAL) AS y
  FROM bonds
  WHERE Price IS NOT NULL
  ORDER BY d
)
SELECT
  d,
  -8.0 * ((y - LAG(y) OVER (ORDER BY d)) / 100.0) AS p
FROM raw;


-- === 3) DAILY RETURNS CALCULATION ===
-- Formula: returns[i] = (price[i] / price[i-1]) - 1

-- Telefonica
DROP VIEW IF EXISTS r_telefonica;
CREATE VIEW r_telefonica AS
SELECT
  'Telefonica' AS asset,
  d,
  p,
  (p / LAG(p) OVER (ORDER BY d) - 1.0) AS r
FROM v_telefonica_clean;

-- Repsol
DROP VIEW IF EXISTS r_repsol;
CREATE VIEW r_repsol AS
SELECT
  'Repsol' AS asset,
  d,
  p,
  (p / LAG(p) OVER (ORDER BY d) - 1.0) AS r
FROM v_repsol_clean;

-- Inditex
DROP VIEW IF EXISTS r_inditex;
CREATE VIEW r_inditex AS
SELECT
  'Inditex' AS asset,
  d,
  p,
  (p / LAG(p) OVER (ORDER BY d) - 1.0) AS r
FROM v_inditex_clean;

-- Iberdrola
DROP VIEW IF EXISTS r_iberdrola;
CREATE VIEW r_iberdrola AS
SELECT
  'Iberdrola' AS asset,
  d,
  p,
  (p / LAG(p) OVER (ORDER BY d) - 1.0) AS r
FROM v_iberdrola_clean;

-- ACS
DROP VIEW IF EXISTS r_acs;
CREATE VIEW r_acs AS
SELECT
  'ACS' AS asset,
  d,
  p,
  (p / LAG(p) OVER (ORDER BY d) - 1.0) AS r
FROM v_acs_clean;

-- BBVA
DROP VIEW IF EXISTS r_bbva;
CREATE VIEW r_bbva AS
SELECT
  'BBVA' AS asset,
  d,
  p,
  (p / LAG(p) OVER (ORDER BY d) - 1.0) AS r
FROM v_bbva_clean;

-- Amadeus
DROP VIEW IF EXISTS r_amadeus;
CREATE VIEW r_amadeus AS
SELECT
  'Amadeus' AS asset,
  d,
  p,
  (p / LAG(p) OVER (ORDER BY d) - 1.0) AS r
FROM v_amadeus_clean;

-- Gold
DROP VIEW IF EXISTS r_gold;
CREATE VIEW r_gold AS
SELECT
  'Gold Futures' AS asset,
  d,
  p,
  (p / LAG(p) OVER (ORDER BY d) - 1.0) AS r
FROM v_gold_clean;

-- Spain 10Y Bonds
DROP VIEW IF EXISTS r_bonds;
CREATE VIEW r_bonds AS
SELECT
  'Spain 10Y Bonds' AS asset,
  d,
  p,
  p AS r 
FROM v_bonds_clean;



-- === 4) MERGE ALL ASSETS INTO A SINGLE VIEW ===

DROP VIEW IF EXISTS r_all;

CREATE VIEW r_all AS
SELECT * FROM r_telefonica
UNION ALL SELECT * FROM r_repsol
UNION ALL SELECT * FROM r_inditex
UNION ALL SELECT * FROM r_iberdrola
UNION ALL SELECT * FROM r_acs
UNION ALL SELECT * FROM r_bbva
UNION ALL SELECT * FROM r_amadeus
UNION ALL SELECT * FROM r_gold
UNION ALL SELECT * FROM r_bonds;

-- === 5) CALCULATE PERFORMANCE METRICS ===
-- Compute 'Daily Mean' and 'Standard Deviation'. Then, their annual values

DROP VIEW IF EXISTS general_data;

CREATE VIEW general_data AS
WITH base AS (
  SELECT asset AS "Active",
    ROUND(AVG(r), 9) AS "Daily Mean",
    ROUND(SQRT(AVG(r*r) - AVG(r)*AVG(r)), 9) AS "Daily Std. Dev."
  FROM r_all
  WHERE r IS NOT NULL
  GROUP BY asset
)
SELECT
  "Active",
  "Daily Mean",
  "Daily Std. Dev.",
  ROUND("Daily Mean" * 252.0, 9) AS "Annual Mean",
  ROUND("Daily Std. Dev." * SQRT(252.0), 9) AS "Annual Volatility"
FROM base;




