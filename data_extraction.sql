/*
    MLP Photometric Filter Converter (c) by Rik Ghosh, Soham Saha

    MLP Photometric Filter Converter is licensed under a
    Creative Commons Attribution 4.0 International Licesnse.

    You should have received a copy of the license along with this
    work. If not, see https://creativecommons.org/licenses/by/4.0
*/

-- Collect Photometric Filter Data from the MaSTAR Gaia EDR3 table in the SDSS DR 17
-- link: http://skyserver.sdss.org/dr17/SearchTools/SQL/

SELECT
    g.psfmag_1 as u,                    -- SDSS psf filter
    g.psfmag_2 as g,                    -- SDSS psf filter
    g.psfmag_3 as r,                    -- SDSS psf filter
    g.psfmag_4 as i,                    -- SDSS psf filter
    g.psfmag_5 as z,                    -- SDSS psf filter
    g.phot_g_mean_mag as Gaia_G,        -- GAIA filter
    g.phot_bp_mean_mag as bp,           -- GAIA filter
    g.phot_rp_mean_mag as rp            -- GAIA filter
FROM mastar_goodstars_xmatch_gaiaedr3 AS g
WHERE g.psfmag_1 != -999                -- Ignore Erroneous Data
