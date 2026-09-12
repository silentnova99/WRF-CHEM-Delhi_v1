"""WRF-Chem namelist scaffolding for the Delhi domain (Module-2).

Generates ``namelist.input`` and ``namelist.fire_emissions`` from the domain
grid and Module-2 config so the coupled-emulator parameters and the real
WRF-Chem reference configuration stay in sync.  These files are informational
for the offline emulator but follow the real WRF-Chem option names.
"""

from __future__ import annotations

from pathlib import Path

from aqf_delhi.wrf.config import Module2Config


def namelist_input(cfg: Module2Config, grid) -> str:
    d = grid.domain
    return f"""\
 &time_control
   run_days                            = 3,
   start_year                          = 2026,2026,
   start_month                         = 10,10,
   start_day                           = 01,01,
   start_hour                          = 00,00,
   end_year                            = 2026,2026,
   end_month                           = 10,10,
   end_day                             = 04,04,
   end_hour                            = 00,00,
   interval_seconds                    = 10800,
   input_from_file                     = .true.,
   auxinput1_inname                    = 'wrfchemi_d<domain>_<date>',
   history_interval                    = 60,
   frames_per_outfile                  = 1,
   restart                             = .false.,
   io_form_history                     = 2,
   io_form_restart                     = 2,
   io_form_input                       = 2,
   io_form_boundary                    = 2,
   io_form_auxinput1                   = 2,
 /

 &domains
   time_step                           = 24,
   max_dom                             = 1,
   e_we                                = {grid.nx + 4},
   e_sn                                = {grid.ny + 4},
   e_vert                              = 40,
   p_top_requested                     = 1000,
   dx                                  = {d.dx_km * 1000},  dy = {d.dy_km * 1000},
   parent_grid_ratio                   = 1,
   i_parent_start                      = 1,
   j_parent_start                      = 1,
   ztop                                = 22000,
   num_metgrid_levels                  = 38,
   p_top                               = 1000,
 /
 &physics
   mp_physics                          = 8,
   ra_lw_physics                       = 4,  ra_sw_physics = 4,
   radt                                = 30,
   sf_sfclay_physics                   = 1,
   sf_surface_physics                  = 2,
   bl_pbl_physics                      = 1,
   bldt                                = 0,
   cu_physics                          = 5,
   cudt                                = 0,
   num_soil_layers                     = 4,
   isfflx                              = 1,
   icloud                              = 1,
 /
 &fdda
   grid_fdda                           = 1,
   gfdda_inname                        = 'wrffdda_d<domain>',
   gfdda_interval_m                    = 1080,  gfdda_end_h = 0,
/
 &chem
   chem_opt                            = 201,
   kemit_opt                           = 1,
   emis_opt                            = 21,
   emiss_inpt_opt                      = 7,
   fire_emis_opt                       = 17,
   plumerisefire_frq                   = 120,
   plumerisefire_ef                    = 1.0,
   mixchem_above_pbl                   = 1,
   emiss_opt_vol                        = 1,
   aero_ra_fire                        = 2,
   biomass_burn_opt                    = 21,
   gas_fire_inject                      = 1,
   aero_fire_inject                     = 1,
   do_plumerise                        = 1,
   do_vegfire                          = 1,
 /
 &bdy_control
   spec_bdy_width                      = 5,
   spec_zone                           = 1,
   relax_zone                          = 4,
   specified                           = .true.,
   nested                              = .false.,
 /
"""


def namelist_fire_emissions(cfg: Module2Config) -> str:
    return f"""\
 &fire
   fire_date                           = '2026-10-01',
   fire_time                           = '00:00:00',
   fire_emis_opt                        = 17,
   fire_emis_ef_pm25                   = {cfg.emissions.pm25_ef:.4f},
   fire_emis_ef_pm10                   = {cfg.emissions.pm25_ef * cfg.emissions.pm10_ratio:.4f},
   fire_min_frp                         = {cfg.emissions.fire_min_frp_mw:.1f},
   plumerise_max_top_m                  = {cfg.plume.max_top_m:.0f},
   plumerise_sensible_fraction          = {cfg.plume.sensible_fraction:.2f},
   plumerise_entrainment                = {cfg.plume.entrainment_coef:.2f},
 /
"""


def write_namelists(out_dir: Path, cfg: Module2Config, grid) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "namelist.input"
    p2 = out_dir / "namelist.fire_emissions"
    p1.write_text(namelist_input(cfg, grid), encoding="utf-8")
    p2.write_text(namelist_fire_emissions(cfg), encoding="utf-8")
    return [p1, p2]