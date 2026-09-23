/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perigee Viewer
/// Reads ORBIT_DATA.json written by the Perigee engine and renders Earth, a look-ahead line per
/// satellite, and a moving marker for each.  Nothing here touches the orbit math.
///
/// Settings: viewer.toml in the working directory (all keys optional, see that file).
/// Usage:    cargo run                       uses viewer.toml's orbit_file
///           cargo run -- path/to/ORBIT_DATA.json   overrides it
///           SORTED_SATS.json (NORAD IDs, epochs) and ELSET.json (names) are read from the same folder.
///
/// Picking a satellite also draws its ground track on the globe (colors.ground_track; alpha 00 hides it).
/// Mouse:    left-drag orbits the camera, scroll zooms (down to camera.min_distance, closer than the
///           whole globe fits), left-click a satellite isolates it, click empty space (or Esc) clears
///           the selection. With a satellite selected the camera pivots around that satellite (it moves
///           with it; scroll comes in to camera.min_sat_distance); cleared, it pivots around the globe.
///           After a drag the view is yours for camera.manual_hold_seconds, then the calculated view
///           (selected satellite, or the station-facing home) glides back in.
/// Keys:     L live/history, Space play/pause, + / - sim speed, R restart, Esc clear selection.
///           Live locks the sim clock to the system clock.  History replays from the earliest epoch;
///           satellites appear when their own element-set epoch is reached.
/////////////////////////////////////////////////////////////////////////////////////////////////////////
pub mod config;
pub mod data;
use config::{hex, hex_rgba8, Config};
use data::Source;

use bevy::core_pipeline::bloom::Bloom;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::Isometry3d;
use bevy::prelude::*;
use bevy::ui::widget::NodeImageMode;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::PrimaryWindow;
use nalgebra::{Const, Dyn, Matrix6xX, OMatrix};
use perigee_orbit::Propagator;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};


const INFO_BOX_W: f32 = 380.0;
const FOCUS_DIM: f32 = 0.18;
const CROSS_TAGS: usize = 24;     // how many AOS / LOS labels can show at once     // alpha for everything that is not the picked satellite
const INFO_BOX_H: f32 = 520.0;   // approximate, for keeping it on screen

//------------------------------------------------------------------------------------------ resources
#[derive(Resource)]
pub struct Orbits(pub Vec<Matrix6xX<f64>>);

//Local propagation: the engine's RK4 integrator running here, a time slice per frame (None when the
//viewer is reading vectors the engine wrote). PropStatus feeds the HUD while the first pass is running.
#[derive(Resource)]
struct Prop(Option<Propagator>);
#[derive(Resource, Default)]
pub struct PropStatus { pub total: usize, pub done: usize, pub finished: bool, pub started: Option<std::time::Instant> }

/// Everything the viewer said while loading its data, in order (perigee-control shows it on its boot page)
#[derive(Resource, Default)]
pub struct BootLog(pub Vec<String>);

/// perigee-control clears this while one of its own tiles has the keyboard; the viewer then ignores typing
#[derive(Resource)]
pub struct ViewerFocus(pub bool);
impl Default for ViewerFocus { fn default() -> Self { Self(true) } }

/// Read Perigee's files again (after the engine ran). `elsets`: everything, satellites respawned and
/// propagation restarted; otherwise just the ranking and the categories.
#[derive(Event, Clone, Copy, Default)]
pub struct ReloadData { pub elsets: bool }
//Catalog epoch of a satellite whose track is not integrated yet: far in the future, so sat_t < 0 hides it
pub const NOT_YET_JD: f64 = 1.0e12;

//Android reports ~2x density on a 1080p TV; keep the UI in real pixels by scaling it by 1/scale_factor.
//Screen-space positions (from world_to_viewport, in logical px) must then be divided by the UiScale.
fn android_ui_scale(windows: Query<&Window, With<PrimaryWindow>>, mut ui_scale: ResMut<UiScale>) {
    if !cfg!(target_os = "android") { return; }
    let Ok(w) = windows.get_single() else { return };
    let want = 1.0 / w.scale_factor().max(0.5);
    if (ui_scale.0 - want).abs() > 1e-3 { ui_scale.0 = want; println!("ui scale set to {want:.3} (window scale factor {:.3})", w.scale_factor()); }
}

//The Apple II character set (Print Char 21, Kreative Korp, free-use license), used for every piece of text
#[derive(Resource)]
pub struct UiFont(pub Handle<Font>);

//Continent outlines as 3D polylines on the globe (vector_globe mode), in the Earth frame before rotation
#[derive(Resource, Default)]
struct Outlines(Vec<Vec<Vec3>>);

//Per-marker flag set by move_satellites, read by the cross drawer
#[derive(Component)]
struct SatInView(bool);

//Per-column NORAD ID (row 0 of SORTED_SATS.json), epoch as Julian date (row 1),
//and NORAD -> name (from ELSET.json).  All optional.
#[derive(Resource)]
pub struct Catalog {
    pub ids: Vec<Option<u32>>,
    pub epochs: Vec<Option<f64>>,
    pub names: HashMap<u32, String>,
    pub omm: HashMap<u32, serde_json::Value>,              // full Space-Track record (ELSET.json)
    pub transmitters: HashMap<u32, Vec<serde_json::Value>>, // SatNOGS records (NORADs.json)
    pub elsets: Option<OMatrix<f64, Const<9>, Dyn>>,        // the 9-row element set matrix (SORTED_SATS.json)
}

#[derive(Resource)]
#[allow(dead_code)]
struct RealClock { started: std::time::Instant }

#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    History, // replay from the earliest epoch in the data; speed / pause / restart apply
    Live,    // sim clock locked to the system clock; satellites where they are right now
}

#[derive(Resource)]
pub struct Sim {
    pub t: f64,        // sim seconds since jd0
    pub speed: f64,    // sim seconds per real second (History only)
    pub paused: bool,  // History only
    pub t_max: f64,    // seconds from jd0 to the end of the shortest trajectory
    pub jd0: f64,      // absolute time (Julian date) that sim.t = 0 corresponds to
    pub jd_hist0: f64, // earliest epoch in the data: History mode starts here
    pub jd_end: f64,   // end of the shortest trajectory (the engine's t_end)
    pub exhausted: bool, // Live mode has run past the end of the propagated vectors: sim stopped, "UPDATE TLES" shown
}

impl Sim {
    /// Absolute sim time as a Julian date
    pub fn jd(&self) -> f64 { self.jd0 + self.t / 86400.0 }
    fn set_mode(&mut self, mode: Mode) {
        match mode {
            Mode::History => { self.jd0 = self.jd_hist0; self.t = 0.0; }
            Mode::Live => { self.jd0 = now_jd(); self.t = 0.0; }
        }
        self.t_max = ((self.jd_end - self.jd0) * 86400.0).max(0.0);
    }
}

//Every column i of satellite s is at absolute time epoch_s + i * step.
//Turn sim time into "seconds since this satellite's epoch" so all satellites line up in absolute time.
pub fn sat_t(sim: &Sim, cat: &Catalog, i: usize) -> f64 {
    let epoch = cat.epochs.get(i).copied().flatten().unwrap_or(sim.jd0);
    (sim.jd0 - epoch) * 86400.0 + sim.t
}

pub fn now_jd() -> f64 {
    let now = chrono::Utc::now();
    let unix = now.timestamp() as f64 + now.timestamp_subsec_micros() as f64 * 1e-6;
    unix / 86400.0 + 2440587.5
}

fn jd_to_datetime(jd: f64) -> Option<chrono::DateTime<chrono::Utc>> {
    let unix = (jd - 2440587.5) * 86400.0;
    chrono::DateTime::from_timestamp(unix.floor() as i64, 0)
}

//"2026-09-18 23:09:12 EDT / 03:09:12 UTC"  (Eastern follows daylight saving automatically)
fn fmt_utc_est(dt: chrono::DateTime<chrono::Utc>) -> String {
    let est = dt.with_timezone(&chrono_tz::America::New_York);
    format!("{} / {}", est.format("%Y-%m-%d %H:%M:%S %Z"), dt.format("%H:%M:%S UTC"))
}

pub fn jd_to_string(jd: f64) -> String {
    jd_to_datetime(jd).map(fmt_utc_est).unwrap_or_else(|| "----".to_string())
}

//------------------------------------------------------------------------------------------ geodesy
const WGS84_A: f64 = 6378.137;            // km
const WGS84_F: f64 = 1.0 / 298.257223563;

//Greenwich mean sidereal time in radians (Vallado), UTC used as UT1.
pub fn gmst_rad(jd: f64) -> f64 {
    let t = (jd - 2451545.0) / 36525.0;
    let secs = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t * t * t;
    (secs.rem_euclid(86400.0) / 240.0).to_radians()
}

//Geodetic (deg, deg, m) -> ECEF km
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
    let (lat, lon, h) = (lat_deg.to_radians(), lon_deg.to_radians(), alt_m / 1000.0);
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let n = WGS84_A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    [(n + h) * lat.cos() * lon.cos(), (n + h) * lat.cos() * lon.sin(), (n * (1.0 - e2) + h) * lat.sin()]
}

//ECEF km -> geodetic (lat deg, lon deg, alt km), iterative
pub fn ecef_to_geodetic(r: [f64; 3]) -> (f64, f64, f64) {
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let p = (r[0] * r[0] + r[1] * r[1]).sqrt();
    let lon = r[1].atan2(r[0]);
    let mut lat = r[2].atan2(p * (1.0 - e2));
    let mut n = WGS84_A;
    for _ in 0..8 {
        n = WGS84_A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
        lat = (r[2] + e2 * n * lat.sin()).atan2(p);
    }
    let alt = if lat.cos().abs() > 1e-6 { p / lat.cos() - n } else { r[2].abs() - n * (1.0 - e2) };
    (lat.to_degrees(), lon.to_degrees(), alt)
}

//Inertial (TEME) km -> ECEF km by rotating about the pole by GMST
pub fn eci_to_ecef(r: [f64; 3], gmst: f64) -> [f64; 3] {
    let (c, s) = (gmst.cos(), gmst.sin());
    [c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]]
}

//Azimuth (deg from north, clockwise), elevation (deg), range (km) from a station to an ECEF point
pub fn look_angles(sta: [f64; 3], lat_deg: f64, lon_deg: f64, tgt: [f64; 3]) -> (f64, f64, f64) {
    let d = [tgt[0] - sta[0], tgt[1] - sta[1], tgt[2] - sta[2]];
    let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
    let (sl, cl, so, co) = (lat.sin(), lat.cos(), lon.sin(), lon.cos());
    let e = -so * d[0] + co * d[1];
    let n = -sl * co * d[0] - sl * so * d[1] + cl * d[2];
    let u = cl * co * d[0] + cl * so * d[1] + sl * d[2];
    let range = (e * e + n * n + u * u).sqrt();
    let az = e.atan2(n).to_degrees().rem_euclid(360.0);
    let el = (u / range).asin().to_degrees();
    (az, el, range)
}

fn fmt_latlon(lat: f64, lon: f64) -> String {
    format!("{:.4}{}  {:.4}{}", lat.abs(), if lat >= 0.0 { "N" } else { "S" }, lon.abs(), if lon >= 0.0 { "E" } else { "W" })
}

//One entry of SATELLITE_RANKS.json written by Perigee (only the fields the viewer needs)
#[derive(Deserialize, Clone, Debug)]
pub struct RankEntry {
    pub rank: usize,
    pub score: f64,
    pub norad_id: u32,
    pub name: String,
    pub in_progress: bool,
    pub minutes_until_aos: f64,
    pub aos_local: String,
    pub los_local: String,
    pub duration_min: f64,
    pub max_el_deg: f64,
    #[serde(default)] pub minutes_left: f64,
    #[serde(default)] pub el_now_deg: f64,
    #[serde(default)] pub duration_term: f64,
    #[serde(default)] pub elevation_term: f64,
    #[serde(default)] pub transmitter_term: f64,
    #[serde(default)] pub freshness_term: f64,
    pub pass: PassRef,
}
#[derive(Deserialize, Clone, Debug)]
pub struct PassRef {
    pub column: usize,
    #[serde(default)] pub aos_jd: f64,
    #[serde(default)] pub los_jd: f64,
    #[serde(default)] pub max_el_jd: f64,
}

#[derive(Deserialize, Clone, Debug, Default)]
struct Weights { duration: f64, elevation: f64, transmitter: f64, freshness: f64 }

//Newer SATELLITE_RANKS.json: settings + weights + entries. Older files are a bare array of entries.
#[derive(Deserialize, Debug)]
struct RankReport {
    #[serde(default)] generated_local: String,
    #[serde(default)] mask_deg: f64,
    #[serde(default)] horizon_min: f64,
    #[serde(default)] weights: Option<Weights>,
    #[serde(default)] station_name: String,
    #[serde(default)] station_lat_deg: f64,
    #[serde(default)] station_lon_deg: f64,
    #[serde(default)] region: Option<RegionName>,
    entries: Vec<RankEntry>,
}
#[derive(Deserialize, Debug, Clone)]
struct RegionName { #[serde(default)] name: String }

#[derive(Resource, Default)]
pub struct Ranks {
    pub entries: Vec<RankEntry>,
    pub columns: HashSet<usize>,
    pub columns_ordered: Vec<usize>,
    weights: Option<Weights>,
    pub generated_local: String,
    pub mask_deg: f64,
    pub horizon_min: f64,
    pub station: Option<(String, f64, f64)>,
    pub region_name: Option<String>,
}

#[derive(Resource, Default)]
struct ScoreOpen(bool);

//Sky windows and which one is selected; `sent` tracks what Perigee was last told
#[derive(Resource, Default)]
struct Regions { list: Vec<config::Region>, current: usize, sent: Option<usize> }

//The VIEW drop-down: open/closed and the highlighted option (remote / arrow keys)
#[derive(Resource, Default)]
struct RegionMenu { open: bool, highlight: usize }

//TYPE drop-down: satellite categories from Perigee's CATEGORIES.json (WEATHER, MILITARY, ...), same shape
//as the VIEW menu. TypeFilter: None = every type, Some(k) = only category k. Menu rows: 0 = ALL, k + 1 = category k.
#[derive(Resource, Default)]
struct TypeMenu { open: bool, highlight: usize }
#[derive(Resource, Default)]
struct TypeFilter(Option<usize>);
#[derive(Resource, Default)]
struct Categories { names: Vec<String>, counts: Vec<usize>, members: Vec<Vec<bool>>, total: usize }
impl Categories {
    fn allows(&self, filter: &TypeFilter, col: usize) -> bool {
        match filter.0 { None => true, Some(k) => self.members.get(k).and_then(|m| m.get(col)).copied().unwrap_or(false) }
    }
    fn label(&self, filter: &TypeFilter) -> String {
        match filter.0 { None => "ALL".into(), Some(k) => self.names.get(k).cloned().unwrap_or_else(|| "ALL".into()) }
    }
    fn rows(&self) -> usize { self.names.len() + 1 }
    //Menu row -> filter (row 0 and anything out of range mean ALL)
    fn filter_for_row(&self, row: usize) -> Option<usize> { if row == 0 || row > self.names.len() { None } else { Some(row - 1) } }
}
#[derive(Component)] struct TypeOption(usize);   // drop-down row
#[derive(Component)] struct TypeHeader;          // the TYPE: ... button label
#[derive(Component)] struct TypeList;            // container of the option rows

//Explore mode: every explore_seconds the camera glides to a random satellite (ranked or not) and the
//rest of the sky dims around it. Toggled with X (PC) or REWIND (remote); off returns to the ranking view.
#[derive(Resource, Default)]
struct Explore { on: bool, next: f64, rng: u64, recent: Vec<usize>, panel_was_hidden: bool }
impl Explore {
    //xorshift64: plenty for picking satellites, and no crate needed
    fn rand(&mut self) -> u64 {
        if self.rng == 0 { self.rng = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x9E3779B97F4A7C15) | 1; }
        let mut x = self.rng; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.rng = x; x
    }
}

//Turn explore on or off: hides the panel while exploring (the globe centres itself), restores it after
fn set_explore(on: bool, now: f64, explore: &mut Explore, sel: &mut Selected, panel_hidden: &mut PanelHidden, panel: &mut Query<&mut Visibility, With<RankPanel>>) {
    if explore.on == on { return; }
    explore.on = on;
    if on {
        explore.panel_was_hidden = panel_hidden.0;
        explore.next = now;                    // first pick right away
        panel_hidden.0 = true;
    } else {
        panel_hidden.0 = explore.panel_was_hidden;
        sel.0 = None;
    }
    for mut v in panel.iter_mut() { *v = if panel_hidden.0 { Visibility::Hidden } else { Visibility::Inherited }; }
}

//Where tracks cross the edge of the view this frame: (world position, entering?) for the AOS / LOS tags
#[derive(Resource, Default)]
struct Crossings(Vec<(Vec3, bool)>);

#[derive(Component)]
struct CrossTag(usize);     // pooled screen labels for the crossings

#[derive(Component)]
struct ExhaustedBox;        // the centred UPDATE TLES notice
#[derive(Component)]
struct ExhaustedCursor;     // its blinking block cursor

//Shared with the remote poller thread: after a view change, poll quickly until a ranking for that
//region name arrives (or the deadline passes)
#[derive(Resource, Clone, Default)]
struct RegionWanted(std::sync::Arc<std::sync::Mutex<Option<(String, std::time::Instant)>>>);

//Ranking panel hidden with P; kept here so the panel stays hidden through its rebuilds
#[derive(Resource, Default)]
struct PanelHidden(bool);

//Info box dragging: once dragged it stays put (pinned) until FOLLOW is pressed or the selection changes
#[derive(Resource, Default)]
struct InfoDrag { dragging: bool, grab_offset: Vec2, pinned: Option<Vec2>, pos: Vec2 }

#[derive(Resource, Default)]
struct RankedOnly(bool);

#[derive(Resource)]
pub struct DataSource(pub std::sync::Arc<Source>);

#[derive(Resource)]
struct RemoteRanks(std::sync::Mutex<std::sync::mpsc::Receiver<String>>);

//Runs "perigee rank" on a timer so the ranking keeps up with the clock
#[derive(Resource)]
struct Rerank { next: f64, child: Option<std::process::Child>, dir: std::path::PathBuf }

//Polls SATELLITE_RANKS.json's modified time so a fresh `perigee rank` shows up without a relaunch
#[derive(Resource)]
struct RankWatch { path: std::path::PathBuf, last_modified: Option<std::time::SystemTime>, next_check: f64, force: bool }

#[derive(Resource, Default)]
struct Search { active: bool, query: String, results: Vec<usize>, highlight: usize }

//Row cursor for the ranking panel, driven by the TV remote's D-pad (or arrow keys on the desktop)
#[derive(Resource, Default)]
struct RowCursor(Option<usize>);

#[derive(Resource, Default)]
pub struct Selected(pub Option<usize>);

#[derive(Resource, Default)]
struct DragState { press_pos: Option<Vec2>, moved: f32 }

//------------------------------------------------------------------------------------------ components
#[derive(Component)]
struct Satellite(usize);

#[derive(Component)]
struct Earth;

#[derive(Component)]
struct StationDot;

//Gizmo group with a thicker line for the coastlines and the station marker
#[derive(Default, Reflect, GizmoConfigGroup)]
struct BoldLines;

//Camera fly-to state: where the glide started, how far along it is, and the view to return to
//hold: seconds left on a manual drag before the calculated view takes over again (camera.manual_hold_seconds)
#[derive(Default)]
struct CamFly { from: (f32, f32, f32), from_pivot: Vec3, t: f32, home: Option<(f32, f32, f32)>, selected: bool, user_zoom: bool, hold: f32 }

#[derive(Component)]
struct RegionOption(usize); // drop-down row

#[derive(Component)]
struct RegionHeader;        // the VIEW: ... button label

#[derive(Component)]
struct RegionList;          // container of the option rows

//Marker materials: swapped per satellite depending on whether it is above the station's elevation mask.
#[derive(Resource)]
struct MarkerMats { mesh: Handle<Mesh>, normal: Handle<StandardMaterial>, in_view: Handle<StandardMaterial>, selected: Handle<StandardMaterial> }

#[derive(Resource, Default)]
struct InView { count: usize }

#[derive(Component)]
//pivot: the point the camera orbits and looks at: the globe's centre, or the selected satellite
struct OrbitCamera { yaw: f32, pitch: f32, distance: f32, pivot: Vec3 }

#[derive(Component)]
struct HudText;

#[derive(Component)]
struct RankTag(usize);   // 1..=3

#[derive(Component)]
struct InfoBox;

#[derive(Component)]
struct InfoTitle;

#[derive(Component)]
struct InfoFollowLabel;

//Every value slot in the info box; the update system fills them by id each frame
#[derive(Component, Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Field {
    IntlId, ObjType, Country, Launched, Site,
    SubPoint, Altitude, Speed, Period,
    AzEl, Range, RangeRate, Status,
    Incl, Raan, ArgP, TrueAnom, Ecc, PeriApo,
    Downlink, Mode, Doppler, TxCount,
    EpochAge, Bstar, MeanMotion,
    Rank, Pass, Left,
}


#[derive(Component)]
struct RankPanel;

#[derive(Component)]
struct ScoreDetail;

#[derive(Component)]
struct SearchBox;

#[derive(Component)]
struct SearchResults;

#[derive(Component)]
struct RankRow(usize);   // column of the satellite this row selects

#[derive(Component)]
struct RankRowIndex(usize);   // position in the panel, for the D-pad cursor

#[derive(Component, Clone, Copy)]
enum ButtonAction { ToggleMode, TogglePause, Slower, Faster, Restart, Clear, ToggleRanked, OpenSearch, ToggleScore, FollowSat, RegionMenu, TypeMenu }

//------------------------------------------------------------------------------------------ main
//IP geolocation via ipinfo.io (what the radar widget uses). Result is cached next to viewer.toml so an
//offline launch still has a position. Returns (name, lat, lon).
fn locate_by_ip() -> Option<(String, f64, f64)> {
    const CACHE: &str = ".station_cache.json";
    let parse = |txt: &str| -> Option<(String, f64, f64)> {
        let v: serde_json::Value = serde_json::from_str(txt).ok()?;
        let loc = v["loc"].as_str()?;
        let (lat, lon) = loc.split_once(',')?;
        let city = v["city"].as_str().unwrap_or("").to_string();
        let region = v["region"].as_str().unwrap_or("").to_string();
        let name = match (city.is_empty(), region.is_empty()) {
            (false, false) => format!("{city}, {region}"),
            (false, true) => city,
            _ => "IP LOCATION".to_string(),
        };
        Some((name, lat.trim().parse().ok()?, lon.trim().parse().ok()?))
    };
    match ureq::get("https://ipinfo.io/json").timeout(std::time::Duration::from_secs(5)).call() {
        Ok(resp) => {
            let txt = resp.into_string().ok()?;
            let r = parse(&txt);
            if r.is_some() { let _ = std::fs::write(CACHE, &txt); }
            r
        }
        Err(e) => {
            eprintln!("ipinfo.io lookup failed ({e}); trying cached position");
            std::fs::read_to_string(CACHE).ok().and_then(|t| parse(&t))
        }
    }
}

//Embedded so the TV build needs no files on disk
const UI_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/PrintChar21.ttf");
const COUNTRIES_GEOJSON: &str = include_str!("../assets/countries.geojson");

#[bevy_main]
pub fn main() {
    build_app("viewer.toml", None).run();
}

/// Everything main() does except running: load the settings and the data, then build the Bevy App with
/// the whole viewer in it. perigee-control calls this and adds its own windows and systems before running.
/// `config_path` is the desktop settings file (the Android build ignores it and uses the embedded one);
/// `orbit_file` replaces the file's [data] orbit_file (and so the folder Perigee's outputs are read from).
/// A command-line argument still wins over both, as it always has.
pub fn build_app(config_path: &str, orbit_file: Option<&str>) -> App {
    #[cfg(target_os = "android")]
    let _ = (config_path, orbit_file);
    //Desktop reads viewer.toml next to the binary; the TV build carries it embedded and overrides what
    //cannot apply there (no files, no IP lookup, no perigee binary to run)
    #[cfg(not(target_os = "android"))]
    let mut cfg = Config::load(config_path);
    #[cfg(target_os = "android")]
    let mut cfg: Config = toml::from_str(include_str!("../viewer.toml")).unwrap_or_default();
    #[cfg(target_os = "android")]
    {
        cfg.data.source = "remote".into();
        cfg.data.rerank_command = String::new();
        cfg.station.auto_locate = false;
    }
    #[cfg(target_os = "android")]
    {
        //A passive display must not let the Fire TV's idle screensaver take over
        if let Some(app) = bevy::window::ANDROID_APP.get() {
            app.set_window_flags(android_activity::WindowManagerFlags::KEEP_SCREEN_ON, android_activity::WindowManagerFlags::empty());
        }
        //Live settings from the PC (tv/viewer-tv.toml): no rebuild needed to tune the TV
        let boot = make_source(&cfg);
        match boot.fetch_config() {
            Some(txt) => match toml::from_str::<Config>(&txt) {
                Ok(c) => { cfg = c; println!("config fetched from the PC"); }
                Err(e) => eprintln!("PC config rejected: {e}; using the embedded one"),
            },
            None => println!("no PC config; using the embedded one"),
        }
        cfg.data.source = "remote".into();
        cfg.data.rerank_command = String::new();
        cfg.station.auto_locate = false;
    }
    if let Some(o) = orbit_file { cfg.data.orbit_file = o.to_string(); }
    if cfg.data.source.eq_ignore_ascii_case("remote") { cfg.data.rerank_command = String::new(); }   // the PC's timers re-rank

    let source = make_source(&cfg);
    let mut log: Vec<String> = Vec::new();
    say(&mut log, format!("data source: {}", source.describe()));
    //The TV keeps retrying until the PC answers; the desktop fails fast so a bad path is obvious.
    let data = loop {
        match load_data(&cfg, &source, &mut log) {
            Ok(d) => break d,
            Err(e) if cfg!(target_os = "android") => { eprintln!("load failed: {e}; retrying in 5 s"); std::thread::sleep(std::time::Duration::from_secs(5)); }
            Err(e) => panic!("{e}"),
        }
    };
    let n_sats = data.orbits.len();
    let DataSet { propagator, orbits, catalog, ranks, categories, jd_hist0, jd_end } = data;

    //Station: IP lookup on the desktop if asked; on the TV (or any remote source) whatever Perigee ranked for
    if cfg.data.source == "remote" {
        if let Some((name, lat, lon)) = &ranks.station {
            cfg.station.name = name.clone(); cfg.station.lat_deg = *lat; cfg.station.lon_deg = *lon;
        }
    } else if cfg.station.auto_locate {
        match locate_by_ip() {
            Some((name, lat, lon)) => {
                println!("station located by IP: {name}  {}", fmt_latlon(lat, lon));
                cfg.station.name = name; cfg.station.lat_deg = lat; cfg.station.lon_deg = lon;
            }
            None => println!("auto_locate failed; using station {}", fmt_latlon(cfg.station.lat_deg, cfg.station.lon_deg)),
        }
    }

    //Sky windows: from config, FULL SKY floor follows the elevation mask; the launch selection by name
    let mut regions = Regions { list: cfg.regions.clone(), current: 0, sent: None };
    if regions.list.is_empty() { regions.list.push(config::Region { el_min: cfg.station.elevation_mask_deg, ..config::Region::default() }); }
    for r in regions.list.iter_mut() { if r.full_azimuth() && r.el_max >= 90.0 && r.name.eq_ignore_ascii_case("FULL SKY") { r.el_min = cfg.station.elevation_mask_deg; } }
    if let Some(i) = regions.list.iter().position(|r| r.name.eq_ignore_ascii_case(&cfg.station.region)) { regions.current = i; }
    //If Perigee already ranked for one of our regions, start on that one so the list matches the drawing
    if cfg.station.region.is_empty() { if let Some(name) = &ranks.region_name { if let Some(i) = regions.list.iter().position(|r| r.name.eq_ignore_ascii_case(name)) { regions.current = i; } } }
    //If Perigee's last ranking was already for this region there is nothing to send; otherwise apply_region sends it
    let already = ranks.region_name.as_deref().map_or(false, |n| regions.list.get(regions.current).map_or(false, |r| r.name.eq_ignore_ascii_case(n)));
    regions.sent = if already { Some(regions.current) } else { None };

    let rank_watch = RankWatch {
        last_modified: match &source { Source::Files { dir, .. } => std::fs::metadata(dir.join("SATELLITE_RANKS.json")).and_then(|m| m.modified()).ok(), _ => None },
        path: match &source { Source::Files { dir, .. } => dir.join("SATELLITE_RANKS.json"), _ => std::path::PathBuf::new() },
        next_check: 0.0, force: false,
    };
    let rerank_dir = match &source { Source::Files { dir, .. } => dir.clone(), _ => std::path::PathBuf::from(".") };
    let source = std::sync::Arc::new(source);
    //Remote rankings arrive from a background thread so a slow network never stalls a frame
    let region_wanted = RegionWanted::default();
    let remote_rx = spawn_remote_rank_poller(source.clone(), cfg.data.ranks_poll_seconds, region_wanted.0.clone());

    let mode = if cfg.sim.start_mode.eq_ignore_ascii_case("history") { Mode::History } else { Mode::Live };
    let mut sim = Sim { t: 0.0, speed: cfg.sim.start_speed, paused: false, t_max: 0.0, jd0: 0.0, jd_hist0, jd_end, exhausted: false };
    sim.set_mode(mode);
    say(&mut log, format!("history window: {} -> {}", jd_to_string(jd_hist0), jd_to_string(jd_end)));
    say(&mut log, format!("live window:    {} -> {} ({:.1} h)", jd_to_string(now_jd()), jd_to_string(jd_end), (jd_end - now_jd()) * 24.0));

    let mut app = App::new();
    app.add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin)
        .add_plugins(bevy::diagnostic::LogDiagnosticsPlugin { wait_duration: std::time::Duration::from_secs(10), ..default() })
        .add_plugins({
            let present_mode = match cfg.perf.present_mode.to_lowercase().as_str() {
                "novsync" | "immediate" => bevy::window::PresentMode::AutoNoVsync,
                "mailbox" => bevy::window::PresentMode::Mailbox,
                "fifo" => bevy::window::PresentMode::Fifo,
                _ => bevy::window::PresentMode::AutoVsync,
            };
            let plugins = DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "PERIGEE // orbit view".into(),
                    present_mode,
                    ..default()
                }),
                ..default()
            });
            if cfg.perf.pipelined_rendering { plugins.build() }
            else { plugins.build().disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>() }
        })
        .insert_resource(ClearColor(hex(&cfg.colors.space)))
        .insert_resource(cfg)
        .insert_resource(Orbits(orbits))
        .insert_resource(Prop(propagator))
        .insert_resource(PropStatus { total: n_sats, ..default() })
        .insert_resource(catalog)
        .insert_resource(RealClock { started: std::time::Instant::now() })
        .insert_resource(sim)
        .insert_resource(mode)
        .init_resource::<Selected>()
        .init_resource::<DragState>()
        .init_resource::<InView>()
        .init_resource::<RankedOnly>()
        .init_resource::<ScoreOpen>()
        .init_resource::<RegionMenu>()
        .insert_resource(categories)
        .init_resource::<TypeFilter>()
        .init_resource::<TypeMenu>()
        .init_resource::<Crossings>()
        .init_resource::<Explore>()
        .init_gizmo_group::<BoldLines>()
        .insert_resource(regions)
        .init_resource::<PanelHidden>()
        .init_resource::<InfoDrag>()
        .init_resource::<Search>()
        .init_resource::<RowCursor>()
        .insert_resource(ranks)
        .insert_resource(rank_watch)
        .insert_resource(DataSource(source))
        .insert_resource(RemoteRanks(std::sync::Mutex::new(remote_rx)))
        .insert_resource(region_wanted)
        .insert_resource(Rerank { next: 5.0, child: None, dir: rerank_dir })
        .init_resource::<Outlines>()
        .insert_resource(BootLog(log))
        .init_resource::<ViewerFocus>()
        .add_event::<ReloadData>()
        .add_systems(Update, reload_data.before(move_satellites))
        .add_systems(Startup, (load_font, setup_bold_lines, setup_scene, setup_hud, setup_fx).chain())
        .add_systems(
            Update,
            (
                android_ui_scale, search_input, keyboard, remote_controls, buttons, rank_rows, region_options, apply_region, watch_ranks, auto_rerank, refresh_ranks_live, explore_tick, update_exhausted,
                advance_time, move_satellites, spin_earth, spin_markers, orbit_camera, pick_satellite,
            ),
        )
        .add_systems(Update, propagate_tick.before(move_satellites))
        .add_systems(Update, (type_options, apply_type))
        .add_systems(
            Update,
            (
                draw_orbits, draw_reticles, draw_rank_rings, draw_vector_globe, draw_cross_markers, update_hud, update_search_ui,
                update_score_detail, drag_info_box, update_info_box, update_rank_tags, update_cross_tags,
            ).after(move_satellites),
        );
    app
}

fn say(log: &mut Vec<String>, s: String) { println!("{s}"); log.push(s); }

/// What a load of Perigee's files produces: the tracks (placeholders while propagating locally), the
/// propagator itself, the catalog, the ranking, the categories and the two time windows.
struct DataSet { propagator: Option<Propagator>, orbits: Vec<Matrix6xX<f64>>, catalog: Catalog, ranks: Ranks, categories: Categories, jd_hist0: f64, jd_end: f64 }

/// Read Perigee's files (or the server's copies) and build everything the viewer keeps about them.
/// Local propagation asks for the raw element sets (88 KB) and integrates them here, a slice per
/// frame; if the source cannot provide them (old server), the engine's vectors are loaded as before.
/// Used at launch and again by `ReloadData` after the engine wrote new files.
fn load_data(cfg: &Config, source: &Source, log: &mut Vec<String>) -> Result<DataSet, String> {
    let step = cfg.data.step_seconds;
    let mut propagator: Option<Propagator> = None;
    let mut loaded = if cfg.data.local_propagation {
        match source.load_elsets() {
            Ok((coe, l)) => {
                let now = now_jd();
                let threads = if cfg.data.propagation_threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { cfg.data.propagation_threads };
                say(log, format!("local propagation: {} element sets, {:.0} ms per frame on {} threads, keep {:.0} min back, {:.0} h ahead",
                         coe.ncols(), cfg.data.propagation_budget_ms, threads, cfg.data.keep_before_min, cfg.data.propagate_ahead_h));
                propagator = Some(Propagator::new(&coe, step, now - cfg.data.keep_before_min / 1440.0, now + cfg.data.propagate_ahead_h / 24.0));
                //Placeholder tracks: one column (the state at the element epoch) so every system has something
                //to index. The satellite stays hidden until propagate_tick publishes its real track.
                let orbits = (0..coe.ncols()).map(|c| Matrix6xX::from_columns(&[perigee_orbit::state_from_coe(&coe, c)])).collect();
                data::Loaded { orbits, ..l }
            }
            Err(e) => { eprintln!("local propagation unavailable ({e}); loading the engine's vectors"); source.load()? }
        }
    } else { source.load()? };
    let orbits = std::mem::take(&mut loaded.orbits);
    let longest = orbits.iter().map(|m| (m.ncols() - 1) as f64 * step).fold(0.0, f64::max);
    if propagator.is_none() { say(log, format!("loaded {} orbits, longest {:.1} h", orbits.len(), longest / 3600.0)); }

    let mut catalog = load_catalog(loaded.sorted_sats.as_deref(), loaded.elset.as_deref(), loaded.transmitters.as_deref(), orbits.len());
    //Nothing is integrated yet: park every epoch in the future so the markers stay hidden until their track lands
    if propagator.is_some() { for e in catalog.epochs.iter_mut() { *e = Some(NOT_YET_JD); } }
    say(log, format!("catalog: {} ids, {} epochs, {} names",
        catalog.ids.iter().flatten().count(), catalog.epochs.iter().flatten().count(), catalog.names.len()));
    let ranks = parse_ranks(loaded.ranks.as_deref(), orbits.len(), cfg.data.top_ranked);
    say(log, format!("ranks: {} entries", ranks.entries.len()));
    let categories = load_categories(loaded.categories.as_deref(), &catalog.ids);
    say(log, format!("categories: {}", if categories.names.is_empty() { "none (no CATEGORIES.json: run `perigee categories` on the PC)".to_string() }
             else { categories.names.iter().zip(&categories.counts).map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", ") }));

    //Two windows over the same data:
    //  History: from the earliest epoch to the end of the shortest trajectory (satellites appear at their epoch)
    //  Live:    from the system clock to that same end
    let (jd_hist0, jd_end) = if propagator.is_some() {
        //Local propagation: history from keep_before_min ago, data good to the propagation target (and kept there)
        let now = now_jd();
        (now - cfg.data.keep_before_min / 1440.0, now + cfg.data.propagate_ahead_h / 24.0)
    } else if catalog.epochs.iter().any(Option::is_some) {
        let earliest = catalog.epochs.iter().flatten().cloned().fold(f64::MAX, f64::min);
        let jd_end = orbits.iter().zip(&catalog.epochs)
            .filter_map(|(m, e)| e.map(|e| e + (m.ncols() - 1) as f64 * step / 86400.0))
            .fold(f64::MAX, f64::min);
        (earliest, jd_end)
    } else {
        (0.0, longest / 86400.0)   // no epochs: fall back to "column i = sim second i*step" for everyone
    };
    Ok(DataSet { propagator, orbits, catalog, ranks, categories, jd_hist0, jd_end })
}

/// `ReloadData` arrived: read the files again. With `elsets` every satellite marker is respawned for the
/// new catalog, the propagator starts over and the pick is cleared; without it only the ranking panel
/// and the categories change. Lines go to stdout and to BootLog like the launch messages.
fn reload_data(
    mut ev: EventReader<ReloadData>, mut commands: Commands, cfg: Res<Config>, source: Res<DataSource>, mode: Res<Mode>,
    mut orbits: ResMut<Orbits>, mut prop: ResMut<Prop>, mut status: ResMut<PropStatus>, mut cat: ResMut<Catalog>,
    mut ranks: ResMut<Ranks>, mut cats: ResMut<Categories>, mut sim: ResMut<Sim>, mut sel: ResMut<Selected>, mut log: ResMut<BootLog>,
    scene: (Res<MarkerMats>, Query<Entity, With<Satellite>>, Query<Entity, With<RankPanel>>),
    extra: (Res<ScoreOpen>, Res<PanelHidden>, Res<UiFont>, Res<Regions>, ResMut<RankWatch>, ResMut<Explore>, ResMut<Search>, ResMut<TypeFilter>, ResMut<RowCursor>),
) {
    let Some(req) = ev.read().last().copied() else { return };
    let (mats, sats, panel) = scene;
    let (score_open, panel_hidden, ui_font, regions, mut watch, mut explore, mut search, mut tfilter, mut row) = extra;
    let mut lines = Vec::new();
    if req.elsets {
        match load_data(&cfg, &source.0, &mut lines) {
            Ok(d) => {
                for e in &sats { commands.entity(e).despawn_recursive(); }
                for i in 0..d.orbits.len() {
                    commands.spawn((Mesh3d(mats.mesh.clone()), MeshMaterial3d(mats.normal.clone()), Transform::default(), Satellite(i), SatInView(false)));
                }
                *status = PropStatus { total: d.propagator.as_ref().map_or(0, |p| p.len()), ..default() };
                prop.0 = d.propagator;
                *orbits = Orbits(d.orbits); *cat = d.catalog; *ranks = d.ranks; *cats = d.categories;
                sim.jd_hist0 = d.jd_hist0; sim.jd_end = d.jd_end; sim.exhausted = false; sim.set_mode(*mode);
                sel.0 = None; explore.recent.clear(); search.results.clear(); search.highlight = 0; tfilter.0 = None; row.0 = None;
                say(&mut lines, format!("reloaded: {} satellites, ranking {} entries", orbits.0.len(), ranks.entries.len()));
            }
            Err(e) => say(&mut lines, format!("reload failed: {e}")),
        }
    } else {
        let txt = source.0.fetch_ranks();
        *ranks = parse_ranks(txt.as_deref(), orbits.0.len(), cfg.data.top_ranked);
        if let Source::Files { dir, .. } = &*source.0 {
            *cats = load_categories(std::fs::read_to_string(dir.join("CATEGORIES.json")).ok().as_deref(), &cat.ids);
        }
        say(&mut lines, format!("rankings reloaded: {} entries", ranks.entries.len()));
    }
    watch.last_modified = std::fs::metadata(&watch.path).and_then(|m| m.modified()).ok();
    for e in &panel { commands.entity(e).despawn_recursive(); }
    spawn_rank_panel(&mut commands, &cfg, &ranks, &regions, &cats, score_open.0, panel_hidden.0, &ui_font.0);
    log.0.extend(lines);
}

fn load_font(mut commands: Commands, mut fonts: ResMut<Assets<Font>>) {
    let font = Font::try_from_bytes(UI_FONT_BYTES.to_vec()).expect("embedded font");
    commands.insert_resource(UiFont(fonts.add(font)));
}

//Files on the desktop; the cast server on the TV (token + certificate fingerprint are baked in at build time)
fn make_source(cfg: &Config) -> Source {
    if cfg.data.source.eq_ignore_ascii_case("remote") {
        let token = option_env!("PERIGEE_CAST_TOKEN").unwrap_or("").to_string();
        let fp = option_env!("PERIGEE_CAST_CERT_SHA256").unwrap_or("");
        if token.len() < 32 || fp.len() != 64 {
            panic!("remote source needs PERIGEE_CAST_TOKEN and PERIGEE_CAST_CERT_SHA256 at build time (tv/build-android.sh sets them)");
        }
        Source::Remote { base: cfg.data.remote_url.trim_end_matches('/').to_string(), token, cert_sha256: data::hex_to_bytes(fp) }
    } else {
        let orbit_file = std::env::args().nth(1).unwrap_or_else(|| cfg.data.orbit_file.clone());
        let orbit_file = std::path::PathBuf::from(orbit_file);
        let dir = orbit_file.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("."));
        Source::Files { dir, orbit_file }
    }
}

//Remote only: a thread that fetches ranks.json every poll_seconds and hands the text over a channel
fn spawn_remote_rank_poller(source: std::sync::Arc<Source>, poll_seconds: f64,
                            wanted: std::sync::Arc<std::sync::Mutex<Option<(String, std::time::Instant)>>>) -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    if matches!(*source, Source::Remote { .. }) {
        let interval = std::time::Duration::from_secs_f64(poll_seconds.max(5.0));
        let fast = std::time::Duration::from_secs(3);
        std::thread::spawn(move || {
            let mut last = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                let want = wanted.lock().ok().and_then(|w| w.clone());
                let hurry = want.as_ref().map_or(false, |(_, deadline)| std::time::Instant::now() < *deadline);
                if last.elapsed() < if hurry { fast } else { interval } { continue; }
                last = std::time::Instant::now();
                if let Some(txt) = source.fetch_ranks() {
                    //Got the ranking for the region we asked for: back to the normal cadence
                    if let Some((name, _)) = &want {
                        let got = serde_json::from_str::<serde_json::Value>(&txt).ok()
                            .and_then(|v| v.get("region")?.get("name")?.as_str().map(str::to_string));
                        if got.map_or(false, |g| g.eq_ignore_ascii_case(name)) { if let Ok(mut w) = wanted.lock() { *w = None; } }
                    }
                    if tx.send(txt).is_err() { break; }
                }
                if want.is_some() && !hurry { if let Ok(mut w) = wanted.lock() { *w = None; } }
            }
        });
    }
    rx
}

fn load_catalog(sorted_sats_txt: Option<&str>, elset_txt: Option<&str>, transmitters_txt: Option<&str>, n: usize) -> Catalog {

    //Column -> NORAD ID (row 0) and epoch JD (row 1) from the element set matrix
    let elsets = sorted_sats_txt.and_then(|txt| serde_json::from_str::<OMatrix<f64, Const<9>, Dyn>>(txt).ok());
    let (ids, epochs) = match &elsets {
        Some(m) => (
            (0..n).map(|c| if c < m.ncols() { Some(m[(0, c)] as u32) } else { None }).collect(),
            (0..n).map(|c| if c < m.ncols() { Some(m[(1, c)]) } else { None }).collect(),
        ),
        None => (vec![None; n], vec![None; n]),
    };

    //NORAD ID -> name and the whole raw Space-Track record (all values are strings there)
    let mut names = HashMap::new();
    let mut omm = HashMap::new();
    if let Some(txt) = elset_txt {
        if let Ok(records) = serde_json::from_str::<Vec<serde_json::Value>>(txt) {
            for rec in records {
                let id = rec["NORAD_CAT_ID"].as_str().and_then(|s| s.parse::<u32>().ok());
                let name = rec["OBJECT_NAME"].as_str().map(str::to_string);
                if let (Some(id), Some(name)) = (id, name) {
                    names.entry(id).or_insert(name);
                    omm.entry(id).or_insert(rec);
                }
            }
        }
    }

    //NORAD ID -> SatNOGS transmitter records
    let mut transmitters: HashMap<u32, Vec<serde_json::Value>> = HashMap::new();
    if let Some(txt) = transmitters_txt {
        if let Ok(records) = serde_json::from_str::<Vec<serde_json::Value>>(txt) {
            for rec in records {
                if let Some(id) = rec["norad_cat_id"].as_u64() {
                    transmitters.entry(id as u32).or_default().push(rec);
                }
            }
        }
    }
    Catalog { ids, epochs, names, omm, transmitters, elsets }
}

//SATELLITE_RANKS.json text -> Ranks; missing => empty list
fn parse_ranks(text: Option<&str>, n: usize, top: usize) -> Ranks {
    let mut ranks = Ranks::default();
    if let Some(txt) = text {
        let parsed: Result<RankReport, _> = serde_json::from_str::<RankReport>(txt)
            .or_else(|_| serde_json::from_str::<Vec<RankEntry>>(txt).map(|entries| RankReport {
                generated_local: String::new(), mask_deg: 0.0, horizon_min: 0.0, weights: None, entries,
                station_name: String::new(), station_lat_deg: 0.0, station_lon_deg: 0.0, region: None,
            }));
        match parsed {
            Ok(report) => {
                if !report.station_name.is_empty() || report.station_lat_deg != 0.0 {
                    ranks.station = Some((report.station_name.clone(), report.station_lat_deg, report.station_lon_deg));
                }
                ranks.region_name = report.region.map(|r| r.name);
                ranks.weights = report.weights;
                ranks.generated_local = report.generated_local;
                ranks.mask_deg = report.mask_deg;
                ranks.horizon_min = report.horizon_min;
                //Only the best `top` passes are kept: everything downstream (panel, rings, filter) sees just those
                for e in report.entries {
                    if ranks.entries.len() >= top.max(1) { break; }
                    if e.pass.column < n {
                        ranks.columns.insert(e.pass.column);
                        ranks.columns_ordered.push(e.pass.column);
                        ranks.entries.push(e);
                    }
                }
            }
            Err(e) => eprintln!("SATELLITE_RANKS.json: {e}"),
        }
    }
    ranks
}

//Rank 1 = full, 2 and 3 step down; None for anything else
fn top_tier(ranks: &Ranks, col: usize) -> Option<(usize, f32)> {
    ranks.entries.iter().find(|e| e.pass.column == col).and_then(|e| match e.rank {
        1 => Some((1, 1.0)),
        2 => Some((2, 0.72)),
        3 => Some((3, 0.5)),
        _ => None,
    })
}

fn scaled(c: Color, k: f32) -> Color {
    let mut s = c.to_srgba();
    s.alpha *= k;
    Color::Srgba(s)
}

pub fn describe(cat: &Catalog, i: usize) -> String {
    match cat.ids.get(i).copied().flatten() {
        Some(id) => match cat.names.get(&id) {
            Some(name) => format!("{name}   NORAD {id}"),
            None => format!("NORAD {id}"),
        },
        None => format!("track #{i}"),
    }
}

//Engine frame: x, y in the equatorial plane, z toward the north pole.
//Bevy: y is up.  (x, y, z) -> (x, z, -y) is a proper rotation that puts the pole up.
fn to_scene(cfg: &Config, x: f64, y: f64, z: f64) -> Vec3 {
    Vec3::new(x as f32, z as f32, -y as f32) / cfg.scene.km_per_unit as f32
}

fn column(cfg: &Config, m: &Matrix6xX<f64>, c: usize) -> Vec3 {
    to_scene(cfg, m[(0, c)], m[(1, c)], m[(2, c)])
}

//Interpolated inertial position in km at t seconds after this satellite's epoch
pub fn sat_eci_km(cfg: &Config, m: &Matrix6xX<f64>, t: f64) -> [f64; 3] {
    let f = (t / cfg.data.step_seconds).clamp(0.0, (m.ncols() - 1) as f64);
    let c0 = f.floor() as usize;
    let c1 = (c0 + 1).min(m.ncols() - 1);
    let a = f - c0 as f64;
    let mut r = [0.0; 3];
    for k in 0..3 { r[k] = m[(k, c0)] + (m[(k, c1)] - m[(k, c0)]) * a; }
    r
}

//Interpolated inertial velocity in km/s at t seconds after epoch
//Orbital period in seconds from the state at t (vis-viva: a = 1 / (2/r - v^2/mu), T = 2 pi sqrt(a^3/mu))
fn sat_period_s(cfg: &Config, m: &Matrix6xX<f64>, t: f64) -> Option<f64> {
    const MU: f64 = 398600.4418;
    let r = sat_eci_km(cfg, m, t);
    let v = sat_eci_vel(cfg, m, t);
    let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let inv_a = 2.0 / rn - v2 / MU;
    if !(inv_a > 0.0) { return None; }              // not a closed orbit (or bad data)
    let a = 1.0 / inv_a;
    Some(2.0 * std::f64::consts::PI * (a * a * a / MU).sqrt())
}

pub fn sat_eci_vel(cfg: &Config, m: &Matrix6xX<f64>, t: f64) -> [f64; 3] {
    let f = (t / cfg.data.step_seconds).clamp(0.0, (m.ncols() - 1) as f64);
    let c0 = f.floor() as usize;
    let c1 = (c0 + 1).min(m.ncols() - 1);
    let a = f - c0 as f64;
    let mut v = [0.0; 3];
    for k in 0..3 { v[k] = m[(3 + k, c0)] + (m[(3 + k, c1)] - m[(3 + k, c0)]) * a; }
    v
}

//Osculating classical elements from a state vector (Curtis Algorithm 4.2).
//Returns (a km, e, i deg, RAAN deg, arg of perigee deg, true anomaly deg)
fn rv_to_coe(r: [f64; 3], v: [f64; 3]) -> (f64, f64, f64, f64, f64, f64) {
    const MU: f64 = 398600.4418;
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();
    let rn = norm(r); let vn = norm(v); let vr = dot(r, v) / rn;
    let h = cross(r, v); let hn = norm(h);
    let incl = (h[2] / hn).acos();
    let n = cross([0.0, 0.0, 1.0], h); let nn = norm(n);
    let raan = if nn > 1e-12 { let x = (n[0] / nn).acos(); if n[1] >= 0.0 { x } else { 2.0 * std::f64::consts::PI - x } } else { 0.0 };
    let ev = [
        ((vn*vn - MU/rn) * r[0] - rn*vr*v[0]) / MU,
        ((vn*vn - MU/rn) * r[1] - rn*vr*v[1]) / MU,
        ((vn*vn - MU/rn) * r[2] - rn*vr*v[2]) / MU,
    ];
    let e = norm(ev);
    let argp = if nn > 1e-12 && e > 1e-12 {
        let x = (dot(n, ev) / (nn * e)).clamp(-1.0, 1.0).acos();
        if ev[2] >= 0.0 { x } else { 2.0 * std::f64::consts::PI - x }
    } else { 0.0 };
    let nu = if e > 1e-12 {
        let x = (dot(ev, r) / (e * rn)).clamp(-1.0, 1.0).acos();
        if vr >= 0.0 { x } else { 2.0 * std::f64::consts::PI - x }
    } else { 0.0 };
    let a = hn * hn / MU / (1.0 - e * e);
    (a, e, incl.to_degrees(), raan.to_degrees(), argp.to_degrees(), nu.to_degrees())
}

fn sat_position(cfg: &Config, m: &Matrix6xX<f64>, t: f64) -> Vec3 {
    let r = sat_eci_km(cfg, m, t);
    to_scene(cfg, r[0], r[1], r[2])
}

//------------------------------------------------------------------------------------------ scene
//Procedural minimalist Earth: dark base, thin graticule, major lines (equator, prime meridian) brighter.
fn earth_texture(cfg: &Config) -> Image {
    let g = &cfg.earth_grid;
    let (w, h) = (g.texture_width.max(64), (g.texture_width.max(64)) / 2);
    let base = hex_rgba8(&cfg.colors.earth);
    let major_c = hex_rgba8(&cfg.colors.grid_major);
    let minor_c = hex_rgba8(&cfg.colors.grid_minor);
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let lon = x as f32 / w as f32 * 360.0;
            let lat = y as f32 / h as f32 * 180.0;
            let near = |v: f32, step: f32, width: f32| step > 0.0 && ((v / step).round() * step - v).abs() < width;
            let px = if near(lon, g.major_deg, g.major_width) || near(lat, g.major_deg, g.major_width) {
                major_c
            } else if near(lon, g.minor_deg, g.minor_width) || near(lat, g.minor_deg, g.minor_width) {
                minor_c
            } else {
                base
            };
            let i = ((y * w + x) * 4) as usize;
            data[i..i + 4].copy_from_slice(&px);
        }
    }

    //Country outlines from a GeoJSON file (Polygon / MultiPolygon rings of [lon, lat])
    if !g.outline_file.is_empty() && g.outline_width_px > 0 {
        match Ok::<String, ()>(COUNTRIES_GEOJSON.to_string()) {
            Ok(txt) => match serde_json::from_str::<serde_json::Value>(&txt) {
                Ok(geo) => {
                    let col = hex_rgba8(&cfg.colors.outline);
                    let mut plot = |x: i64, y: i64| {
                        let half = g.outline_width_px.max(1) as i64 / 2;
                        for dy in -half..=half { for dx in -half..=half {
                            let (px, py) = ((x + dx).rem_euclid(w as i64), (y + dy).clamp(0, h as i64 - 1));
                            let i = ((py as u32 * w + px as u32) * 4) as usize;
                            data[i..i + 4].copy_from_slice(&col);
                        } }
                    };
                    //u: longitude east of Greenwich, 0..1 ; v: 0 at the north pole (matches Bevy's UV sphere)
                    let to_px = |lon: f64, lat: f64| ((lon.rem_euclid(360.0) / 360.0 * w as f64), ((90.0 - lat) / 180.0 * h as f64));
                    let mut rings = 0usize;
                    for feat in geo["features"].as_array().into_iter().flatten() {
                        let geom = &feat["geometry"];
                        let polys: Vec<&serde_json::Value> = match geom["type"].as_str() {
                            Some("Polygon") => vec![&geom["coordinates"]],
                            Some("MultiPolygon") => geom["coordinates"].as_array().into_iter().flatten().collect(),
                            _ => vec![],
                        };
                        for poly in polys {
                            for ring in poly.as_array().into_iter().flatten() {
                                let pts: Vec<(f64, f64)> = ring.as_array().into_iter().flatten()
                                    .filter_map(|c| Some((c[0].as_f64()?, c[1].as_f64()?))).collect();
                                for pair in pts.windows(2) {
                                    let (x0, y0) = to_px(pair[0].0, pair[0].1);
                                    let (x1, y1) = to_px(pair[1].0, pair[1].1);
                                    if (x1 - x0).abs() > w as f64 / 2.0 { continue; } // crosses the seam
                                    let steps = (x1 - x0).abs().max((y1 - y0).abs()).ceil().max(1.0) as i64;
                                    for k in 0..=steps {
                                        let f = k as f64 / steps as f64;
                                        plot((x0 + (x1 - x0) * f).round() as i64, (y0 + (y1 - y0) * f).round() as i64);
                                    }
                                }
                                rings += 1;
                            }
                        }
                    }
                    println!("earth outlines: {rings} rings from {}", g.outline_file);
                }
                Err(e) => eprintln!("{}: not valid GeoJSON ({e}); no outlines", g.outline_file),
            },
            Err(_) => {}
        }
    }

    Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn setup_bold_lines(cfg: Res<Config>, mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<BoldLines>();
    config.line_width = cfg.scene.outline_line_width.max(1.0);
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    orbits: Res<Orbits>,
    cfg: Res<Config>,
) {
    let sc = &cfg.scene;
    let r_earth = (sc.earth_radius_km / sc.km_per_unit) as f32;

    //Vector globe: continent rings as 3D polylines just above the surface, Earth frame (unrotated)
    if sc.vector_globe && !cfg.earth_grid.outline_file.is_empty() {
        let mut rings: Vec<Vec<Vec3>> = Vec::new();
        {
            if let Ok(geo) = serde_json::from_str::<serde_json::Value>(COUNTRIES_GEOJSON) {
                let rr = (r_earth * 1.004) as f64;
                for feat in geo["features"].as_array().into_iter().flatten() {
                    let geom = &feat["geometry"];
                    let polys: Vec<&serde_json::Value> = match geom["type"].as_str() {
                        Some("Polygon") => vec![&geom["coordinates"]],
                        Some("MultiPolygon") => geom["coordinates"].as_array().into_iter().flatten().collect(),
                        _ => vec![],
                    };
                    for poly in polys {
                        for ring in poly.as_array().into_iter().flatten() {
                            let pts: Vec<Vec3> = ring.as_array().into_iter().flatten().filter_map(|c| {
                                let (lon, lat) = (c[0].as_f64()?.to_radians(), c[1].as_f64()?.to_radians());
                                //ECEF unit vector -> scene axes (x, z, -y); no scale since we set the radius here
                                let (x, y, z) = (lat.cos() * lon.cos() * rr, lat.cos() * lon.sin() * rr, lat.sin() * rr);
                                Some(Vec3::new(x as f32, z as f32, -y as f32))
                            }).collect();
                            if pts.len() > 1 { rings.push(pts); }
                        }
                    }
                }
            }
        }
        println!("vector globe: {} outline rings", rings.len());
        commands.insert_resource(Outlines(rings));
    }

    //Earth frame: rotates with the planet (GMST about the pole). Its local axes are ECEF mapped through
    //to_scene, so anything Earth-fixed (the station dot) is a child placed with to_scene(ecef).
    let earth_mesh = meshes.add(Sphere::new(r_earth).mesh().uv(96, 48));
    let earth_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE, // texture carries the colour
        base_color_texture: Some(images.add(earth_texture(&cfg))),
        perceptual_roughness: 0.95,
        metallic: 0.0,
        unlit: sc.earth_unlit,    // vector-display look: no shading at all
        ..default()
    });
    let st = &cfg.station;
    let sta = geodetic_to_ecef(st.lat_deg, st.lon_deg, st.alt_m);
    let station_mesh = meshes.add(Sphere::new(sc.station_size));
    let station_col = hex(&cfg.colors.station).to_linear();
    let station_mat = materials.add(StandardMaterial {
        base_color: hex(&cfg.colors.station),
        emissive: LinearRgba::new(station_col.red * 3.0, station_col.green * 3.0, station_col.blue * 3.0, 1.0),
        ..default()
    });
    commands.spawn((Transform::default(), Visibility::default(), Earth)).with_children(|frame| {
        //Bevy's UV sphere has its pole on local z and longitude 0 on local +x. Rotating -90 deg about x
        //puts the pole on +y and keeps lon 0 on +x, which is exactly ECEF through to_scene.
        frame.spawn((
            Mesh3d(earth_mesh),
            MeshMaterial3d(earth_mat),
            Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        ));
        let sta_scene = to_scene(&cfg, sta[0], sta[1], sta[2]);
        frame.spawn((
            Mesh3d(station_mesh),
            MeshMaterial3d(station_mat),
            Transform::from_translation(sta_scene),
            StationDot,
        ));

    });

    //Atmosphere shell: soft unlit haze just above the surface (skipped when its colour has alpha 0)
    let atmo = hex(&cfg.colors.atmosphere).to_linear();
    if atmo.alpha > 0.0 { commands.spawn((
        Mesh3d(meshes.add(Sphere::new(r_earth * sc.atmosphere_scale).mesh().uv(64, 32))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: hex(&cfg.colors.atmosphere),
            emissive: LinearRgba::new(atmo.red * 0.3, atmo.green * 0.3, atmo.blue * 0.3, 1.0),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        })),
        Transform::default(),
    )); }

    //Satellite markers: small emissive diamonds (bloom makes them glow)
    let glow = cfg.colors.marker_glow;
    let marker_mesh = meshes.add(Cuboid::new(sc.marker_size, sc.marker_size, sc.marker_size));
    let marker_mat = materials.add(StandardMaterial {
        base_color: hex(&cfg.colors.marker),
        emissive: LinearRgba::new(glow[0], glow[1], glow[2], 1.0),
        ..default()
    });
    let vglow = cfg.colors.marker_in_view_glow;
    let in_view_mat = materials.add(StandardMaterial {
        base_color: hex(&cfg.colors.marker_in_view),
        emissive: LinearRgba::new(vglow[0], vglow[1], vglow[2], 1.0),
        ..default()
    });
    let sglow = cfg.colors.selected_glow;
    let selected_mat = materials.add(StandardMaterial {
        base_color: hex(&cfg.colors.selected),
        emissive: LinearRgba::new(sglow[0], sglow[1], sglow[2], 1.0),
        ..default()
    });
    commands.insert_resource(MarkerMats { mesh: marker_mesh.clone(), normal: marker_mat.clone(), in_view: in_view_mat, selected: selected_mat });
    for i in 0..orbits.0.len() {
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(marker_mat.clone()),
            Transform::default(),
            Satellite(i),
            SatInView(false),
        ));
    }

    //Star field: faint unlit points far away
    let star_mesh = meshes.add(Sphere::new(sc.star_size));
    let star_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.8, 0.9),
        emissive: LinearRgba::new(0.9, 1.0, 1.2, 1.0),
        unlit: true,
        ..default()
    });
    let mut seed: u32 = 0x9E37_79B9;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; (seed as f32 / u32::MAX as f32) * 2.0 - 1.0 };
    for _ in 0..sc.star_count {
        let dir = Vec3::new(rnd(), rnd(), rnd()).normalize_or_zero();
        if dir == Vec3::ZERO { continue; }
        let scale = 0.5 + 0.5 * (rnd() + 1.0) * 0.5;
        commands.spawn((
            Mesh3d(star_mesh.clone()),
            MeshMaterial3d(star_mat.clone()),
            Transform::from_translation(dir * sc.star_distance).with_scale(Vec3::splat(scale)),
        ));
    }

    //Lighting: one hard "sun", faint cool fill so the night side reads as a silhouette
    let l = &cfg.lighting;
    commands.spawn((
        DirectionalLight { illuminance: l.sun_illuminance, color: Color::srgb(1.0, 0.97, 0.9), ..default() },
        Transform::from_xyz(l.sun_direction[0], l.sun_direction[1], l.sun_direction[2]).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(AmbientLight { color: Color::srgb(0.5, 0.7, 1.0), brightness: l.ambient_brightness });

    //Camera with HDR + bloom so the emissive markers and lines glow
    let c = &cfg.camera;
    let hdr = cfg.perf.hdr;
    let bloom = hdr && l.bloom_intensity > 0.0;
    let tonemap = match cfg.perf.tonemapping.to_lowercase().as_str() {
        "none" => Tonemapping::None,
        "reinhard" => Tonemapping::Reinhard,
        "aces" => Tonemapping::AcesFitted,
        _ => Tonemapping::TonyMcMapface,
    };
    let mut cam = commands.spawn((
        Camera3d::default(),
        Camera { hdr, ..default() },
        tonemap,
        Transform::from_xyz(0.0, 0.0, c.start_distance).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera { yaw: 0.0, pitch: c.start_pitch, distance: c.start_distance, pivot: Vec3::ZERO },
    ));
    if bloom { cam.insert(Bloom { intensity: l.bloom_intensity, ..Bloom::NATURAL }); }
    if !cfg.perf.msaa { cam.insert(Msaa::Off); }
}

//------------------------------------------------------------------------------------------ hud
fn setup_hud(mut commands: Commands, cfg: Res<Config>, ranks: Res<Ranks>, regions: Res<Regions>, cats: Res<Categories>, ui_font: Res<UiFont>) {
    let font = ui_font.0.clone();
    let text_c = hex(&cfg.colors.text);

    //Top-left column: the readout, then the detail card for the selected satellite underneath it
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Px(16.0),
            top: Val::Px(14.0),
            width: Val::Percent(55.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(12.0),
            ..default()
        })
        .with_children(|col| {
            col.spawn((
                Text::new(""),
                TextFont { font: font.clone(), font_size: 15.0, ..default() },
                TextColor(text_c),
                HudText,
            ));
        });

    //Info box that follows the selected satellite: title bar, section header strips, label / value rows
    spawn_info_box(&mut commands, &cfg, &font);

    //UPDATE TLES notice: shown when Live time runs past the last propagated vector
    {
        let warn = hex(&cfg.colors.los);
        let text_c = hex(&cfg.colors.text);
        let mut bg = hex(&cfg.colors.space).to_srgba(); bg.alpha = 0.92;
        let advice = if cfg!(target_os = "android") { "PERIGEE MUST RUN ON THE PC, THEN RELAUNCH THIS APP" } else { "RUN PERIGEE, THEN RELAUNCH THE VIEWER" };
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Percent(50.0), top: Val::Percent(50.0),
                    margin: UiRect { left: Val::Px(-300.0), top: Val::Px(-90.0), ..default() },
                    width: Val::Px(600.0),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    row_gap: Val::Px(8.0),
                    padding: UiRect::axes(Val::Px(24.0), Val::Px(18.0)),
                    border: UiRect::all(Val::Px(3.0)),
                    ..default()
                },
                BackgroundColor(Color::Srgba(bg)),
                BorderColor(warn),
                GlobalZIndex(20),
                PickingBehavior::IGNORE,
                Visibility::Hidden,
                ExhaustedBox,
            ))
            .with_children(|b| {
                b.spawn((Text::new("*** ORBIT DATA EXHAUSTED ***"), TextFont { font: font.clone(), font_size: 14.0, ..default() }, TextColor(warn)));
                b.spawn((Text::new("NO PROPAGATED VECTORS PAST THIS TIME. SIMULATION STOPPED."), TextFont { font: font.clone(), font_size: 11.0, ..default() }, TextColor(text_c)));
                b.spawn(Node { flex_direction: FlexDirection::Row, column_gap: Val::Px(4.0), margin: UiRect::vertical(Val::Px(6.0)), ..default() })
                 .with_children(|r| {
                    r.spawn((Text::new("UPDATE TLES"), TextFont { font: font.clone(), font_size: 30.0, ..default() }, TextColor(warn)));
                    r.spawn((Text::new("\u{2588}"), TextFont { font: font.clone(), font_size: 30.0, ..default() }, TextColor(warn), ExhaustedCursor));
                 });
                b.spawn((Text::new(advice), TextFont { font: font.clone(), font_size: 11.0, ..default() }, TextColor(text_c)));
            });
    }

    //Pooled AOS / LOS labels for view-edge crossings (placed each frame, hidden when unused)
    for i in 0..CROSS_TAGS {
        let mut bg = hex(&cfg.colors.space).to_srgba(); bg.alpha = 0.75;
        commands
            .spawn((
                Node { position_type: PositionType::Absolute, left: Val::Px(0.0), top: Val::Px(0.0),
                       padding: UiRect::axes(Val::Px(4.0), Val::Px(0.0)), ..default() },
                BackgroundColor(Color::Srgba(bg)),
                GlobalZIndex(7),
                PickingBehavior::IGNORE,
                Visibility::Hidden,
                CrossTag(i),
            ))
            .with_children(|t| {
                t.spawn((Text::new(""), TextFont { font: font.clone(), font_size: 10.0, ..default() },
                         TextColor(hex(&cfg.colors.aos)), PickingBehavior::IGNORE));
            });
    }

    //Numbered tags that ride beside the top three ranked satellites; tag 0 is the name tag of the pick
    for rank in 0..=3usize {
        let mut bg = hex(&cfg.colors.space).to_srgba(); bg.alpha = 0.8;
        let col = if rank == 0 { hex(&cfg.colors.selected) } else { scaled(hex(&cfg.colors.rank_top), 1.15 - 0.25 * rank as f32) };
        commands
            .spawn((
                Node { position_type: PositionType::Absolute, left: Val::Px(0.0), top: Val::Px(0.0),
                       padding: UiRect::axes(Val::Px(5.0), Val::Px(1.0)), border: UiRect::all(Val::Px(2.0)), ..default() },
                BackgroundColor(Color::Srgba(bg)),
                BorderColor(col),
                GlobalZIndex(8),
                PickingBehavior::IGNORE,
                Visibility::Hidden,
                RankTag(rank),
            ))
            .with_children(|t| {
                t.spawn((Text::new(""), TextFont { font: font.clone(), font_size: 12.0, ..default() },
                         TextColor(col), PickingBehavior::IGNORE));
            });
    }

    spawn_rank_panel(&mut commands, &cfg, &ranks, &regions, &cats, false, false, &font);

    //Bottom-right button bar (mouse only: hidden on the TV)
    commands
        .spawn((if cfg!(target_os = "android") { Visibility::Hidden } else { Visibility::Inherited }, Node {
            position_type: PositionType::Absolute,
            right: Val::Px(16.0),
            bottom: Val::Px(12.0),
            max_width: Val::Percent(58.0),
            flex_wrap: FlexWrap::Wrap,
            justify_content: JustifyContent::FlexEnd,
            column_gap: Val::Px(6.0),
            row_gap: Val::Px(6.0),
            ..default()
        }))
        .with_children(|bar| {
            for (label, action) in [
                ("LIVE / HISTORY", ButtonAction::ToggleMode),
                ("<<", ButtonAction::Slower),
                ("PLAY / PAUSE", ButtonAction::TogglePause),
                (">>", ButtonAction::Faster),
                ("RESTART", ButtonAction::Restart),
                ("CLEAR", ButtonAction::Clear),
                ("RANKED ONLY", ButtonAction::ToggleRanked),
            ] {
                bar.spawn((
                    Button,
                    action,
                    Node {
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                        border: UiRect::all(Val::Px(2.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(hex(&cfg.colors.button)),
                    BorderColor(hex(&cfg.colors.button_border)),
                ))
                .with_children(|b| {
                    b.spawn((Text::new(label.to_uppercase()), TextFont { font: font.clone(), font_size: 12.0, ..default() }, TextColor(text_c)));
                });
            }
        });
}

//Top-right: search box (click or "/" to type), results under it, then the ranking panel.
//Called at startup and again whenever SATELLITE_RANKS.json changes on disk.
//Two deliberate lines per ranking row so it fits a narrow panel
fn rank_row_label(e: &RankEntry) -> String {
    let when = if e.in_progress { " NOW ".to_string() } else { format!("{:4.0}m", e.minutes_until_aos) };
    let aos_hm = e.aos_local.get(11..16).unwrap_or("--:--");
    let los_hm = e.los_local.get(11..16).unwrap_or("--:--");
    let el_now = if e.in_progress { format!("el now {:4.1}  ", e.el_now_deg) } else { String::new() };
    let left = if e.minutes_left > 0.0 { e.minutes_left } else { e.duration_min };
    format!("{:2}  {:.2}  {:<18.18} {:5}  {}\n      {}-{}   {:4.1} MIN LEFT   {}PEAK {:4.1}",
        e.rank, e.score, e.name, e.norad_id, when, aos_hm, los_hm, left, el_now.to_uppercase(), e.max_el_deg)
}

fn spawn_rank_panel(commands: &mut Commands, cfg: &Config, ranks: &Ranks, regions: &Regions, cats: &Categories, score_open: bool, hidden: bool, font: &Handle<Font>) {
    let text_c = hex(&cfg.colors.text);
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(16.0),
                top: Val::Px(14.0),
                width: Val::Percent(34.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                ..default()
            },
            if hidden { Visibility::Hidden } else { Visibility::Inherited },
            RankPanel,
        ))
        .with_children(|col| {
            //Spawned on the TV as well (hidden): skipping it there coincided with the UI not rendering at all
            col.spawn((
                Button,
                ButtonAction::OpenSearch,
                if cfg!(target_os = "android") { Visibility::Hidden } else { Visibility::Inherited },
                Node {
                    padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                    border: UiRect::all(Val::Px(2.0)),
                    ..default()
                },
                BackgroundColor(hex(&cfg.colors.button)),
                BorderColor(hex(&cfg.colors.button_border)),
            ))
            .with_children(|b| {
                b.spawn((Text::new("SEARCH  /  NAME OR NORAD"), TextFont { font: font.clone(), font_size: 12.0, ..default() }, TextColor(text_c), SearchBox));
            });
            col.spawn((
                Text::new(""),
                TextFont { font: font.clone(), font_size: 12.0, ..default() },
                TextColor(hex(&cfg.colors.text_dim)),
                SearchResults,
            ));
            //VIEW drop-down: header button, then the option rows (hidden until opened)
            col.spawn((
                Button,
                ButtonAction::RegionMenu,
                Node { padding: UiRect::axes(Val::Px(8.0), Val::Px(3.0)), border: UiRect::all(Val::Px(2.0)), margin: UiRect::top(Val::Px(6.0)), ..default() },
                BackgroundColor(hex(&cfg.colors.button)),
                BorderColor(hex(&cfg.colors.button_border)),
            ))
            .with_children(|b| {
                b.spawn((Text::new(""), TextFont { font: font.clone(), font_size: 12.0, ..default() }, TextColor(text_c), RegionHeader));
            });
            col.spawn((
                Node { flex_direction: FlexDirection::Column, row_gap: Val::Px(2.0), padding: UiRect::left(Val::Px(12.0)), ..default() },
                Visibility::Hidden,
                RegionList,
            ))
            .with_children(|list| {
                for (i, r) in regions.list.iter().enumerate() {
                    list.spawn((
                        Button,
                        RegionOption(i),
                        Node { padding: UiRect::axes(Val::Px(8.0), Val::Px(2.0)), border: UiRect::all(Val::Px(2.0)), ..default() },
                        BackgroundColor(hex(&cfg.colors.button)),
                        BorderColor(Color::NONE),
                    ))
                    .with_children(|b| {
                        let label = if r.full_azimuth() { format!("{}   EL {:.0}-{:.0}", r.name, r.el_min, r.el_max) }
                                    else { format!("{}   AZ {:.0}-{:.0}  EL {:.0}-{:.0}", r.name, r.az_from, r.az_to, r.el_min, r.el_max) };
                        b.spawn((Text::new(label.to_uppercase()), TextFont { font: font.clone(), font_size: 11.0, ..default() }, TextColor(text_c)));
                    });
                }
            });

            //TYPE drop-down: satellite categories from CATEGORIES.json; ALL first, then each category with its count
            col.spawn((
                Button,
                ButtonAction::TypeMenu,
                Node { padding: UiRect::axes(Val::Px(8.0), Val::Px(3.0)), border: UiRect::all(Val::Px(2.0)), margin: UiRect::top(Val::Px(4.0)), ..default() },
                BackgroundColor(hex(&cfg.colors.button)),
                BorderColor(hex(&cfg.colors.button_border)),
            ))
            .with_children(|b| {
                b.spawn((Text::new(""), TextFont { font: font.clone(), font_size: 12.0, ..default() }, TextColor(text_c), TypeHeader));
            });
            col.spawn((
                Node { flex_direction: FlexDirection::Column, row_gap: Val::Px(2.0), padding: UiRect::left(Val::Px(12.0)), ..default() },
                Visibility::Hidden,
                TypeList,
            ))
            .with_children(|list| {
                let mut labels = vec![format!("ALL   {}", cats.total)];
                labels.extend(cats.names.iter().zip(&cats.counts).map(|(n, c)| format!("{n}   {c}")));
                if cats.names.is_empty() { labels.push("NO CATEGORIES.JSON   RUN `PERIGEE CATEGORIES` ON THE PC".into()); }
                for (i, label) in labels.iter().enumerate() {
                    list.spawn((
                        Button,
                        TypeOption(i),
                        Node { padding: UiRect::axes(Val::Px(8.0), Val::Px(2.0)), border: UiRect::all(Val::Px(2.0)), ..default() },
                        BackgroundColor(hex(&cfg.colors.button)),
                        BorderColor(Color::NONE),
                    ))
                    .with_children(|b| {
                        b.spawn((Text::new(label.to_uppercase()), TextFont { font: font.clone(), font_size: 11.0, ..default() }, TextColor(text_c)));
                    });
                }
            });

            col.spawn((
                Text::new(if ranks.entries.is_empty() { "RANKING  (no SATELLITE_RANKS.json)".to_string() }
                          else if cfg!(target_os = "android") { format!("RANKING  {}  ({} PASSES)   PLAY/PAUSE FILTER   MENU HIDE", ranks.region_name.clone().unwrap_or_else(|| "NEXT 15 MIN".into()).to_uppercase(), ranks.entries.len()) }
                          else { format!("RANKING  {}  ({} PASSES)   K FILTER   P HIDE", ranks.region_name.clone().unwrap_or_else(|| "NEXT 15 MIN".into()).to_uppercase(), ranks.entries.len()) }),
                TextFont { font: font.clone(), font_size: 12.0, ..default() },
                TextColor(text_c),
                Node { margin: UiRect::top(Val::Px(10.0)), ..default() },
            ));
            //Collapsible score section: the weights, and the breakdown for the selected satellite
            col.spawn((
                Button,
                ButtonAction::ToggleScore,
                Node { padding: UiRect::axes(Val::Px(8.0), Val::Px(3.0)), border: UiRect::all(Val::Px(2.0)), ..default() },
                BackgroundColor(hex(&cfg.colors.button)),
                BorderColor(hex(&cfg.colors.button_border)),
            ))
            .with_children(|b| {
                b.spawn((Text::new(match (score_open, cfg!(target_os = "android")) {
                            (true, true) => "SCORE WEIGHTS  [-]  REWIND",
                            (false, true) => "SCORE WEIGHTS  [+]  REWIND",
                            (true, false) => "SCORE WEIGHTS  [-]",
                            (false, false) => "SCORE WEIGHTS  [+]",
                         }),
                         TextFont { font: font.clone(), font_size: 12.0, ..default() }, TextColor(text_c)));
            });
            col.spawn((
                Text::new(""),
                TextFont { font: font.clone(), font_size: 11.0, ..default() },
                TextColor(text_c),
                Node { padding: UiRect::axes(Val::Px(8.0), Val::Px(2.0)), ..default() },
                if score_open { Visibility::Inherited } else { Visibility::Hidden },
                ScoreDetail,
            ));
            for (row_i, e) in ranks.entries.iter().enumerate() {
                let label = rank_row_label(e);
                col.spawn((
                    Button,
                    RankRow(e.pass.column),
                    RankRowIndex(row_i),
                    Node {
                        padding: UiRect::axes(Val::Px(8.0), Val::Px(3.0)),
                        border: UiRect::all(Val::Px(2.0)),
                        ..default()
                    },
                    BackgroundColor(hex(&cfg.colors.button)),
                    BorderColor(hex(&cfg.colors.button_border)),
                ))
                .with_children(|b| {
                    b.spawn((Text::new(label.to_uppercase()), TextFont { font: font.clone(), font_size: 12.0, ..default() }, TextColor(text_c)));
                });
            }
        });

}

//CRT feel: a tiled scanline texture and a vignette laid over the whole window, above every UI element,
//ignored by picking so buttons underneath still work
fn setup_fx(mut commands: Commands, mut images: ResMut<Assets<Image>>, cfg: Res<Config>) {
    let fx = &cfg.fx;
    if fx.scanlines > 0.0 {
        //Tiled UI images cost one quad per tile, so the tile must be wide: 512 px wide x one period tall
        //keeps a 1080p screen at a few thousand quads instead of hundreds of thousands.
        let period = fx.scanline_period_px.max(2);
        let width = 512u32;
        let mut data = vec![0u8; (width * period * 4) as usize];
        let dark = (fx.scanlines.clamp(0.0, 1.0) * 255.0) as u8;
        for x in 0..width { data[(x * 4 + 3) as usize] = dark; }   // first row dark, the rest clear
        let img = Image::new(
            Extent3d { width, height: period, depth_or_array_layers: 1 },
            TextureDimension::D2, data, TextureFormat::Rgba8UnormSrgb, RenderAssetUsages::RENDER_WORLD,
        );
        let mut img = img;
        img.sampler = bevy::image::ImageSampler::nearest();
        commands.spawn((
            ImageNode { image: images.add(img), image_mode: NodeImageMode::Tiled { tile_x: true, tile_y: true, stretch_value: 1.0 }, ..default() },
            Node { position_type: PositionType::Absolute, left: Val::Px(0.0), top: Val::Px(0.0), width: Val::Percent(100.0), height: Val::Percent(100.0), ..default() },
            GlobalZIndex(50),
            PickingBehavior::IGNORE,
        ));
    }
    if fx.vignette > 0.0 {
        //Radial falloff baked into a small texture, stretched over the window
        let n = 64u32;
        let mut data = vec![0u8; (n * n * 4) as usize];
        for y in 0..n { for x in 0..n {
            let dx = (x as f32 + 0.5) / n as f32 - 0.5;
            let dy = (y as f32 + 0.5) / n as f32 - 0.5;
            let r = (dx * dx + dy * dy).sqrt() * 2.0;              // 0 centre .. ~1.41 corners
            let a = ((r - 0.55).max(0.0) / 0.85).clamp(0.0, 1.0);
            let i = ((y * n + x) * 4) as usize;
            data[i + 3] = (a * a * fx.vignette.clamp(0.0, 1.0) * 255.0) as u8;
        } }
        let img = Image::new(
            Extent3d { width: n, height: n, depth_or_array_layers: 1 },
            TextureDimension::D2, data, TextureFormat::Rgba8UnormSrgb, RenderAssetUsages::RENDER_WORLD,
        );
        commands.spawn((
            ImageNode { image: images.add(img), ..default() },
            Node { position_type: PositionType::Absolute, left: Val::Px(0.0), top: Val::Px(0.0), width: Val::Percent(100.0), height: Val::Percent(100.0), ..default() },
            GlobalZIndex(49),
            PickingBehavior::IGNORE,
        ));
    }
}

fn spawn_info_box(commands: &mut Commands, cfg: &Config, font: &Handle<Font>) {
    let accent = hex(&cfg.colors.reticle_sel);
    let text_c = hex(&cfg.colors.text);
    let dim_c = hex(&cfg.colors.text_dim);
    let mut panel_bg = hex(&cfg.colors.space).to_srgba(); panel_bg.alpha = 0.90;
    let mut header_bg = hex(&cfg.colors.button_border).to_srgba(); header_bg.alpha = 0.22;
    let mut title_bg = accent.to_srgba(); title_bg.alpha = 0.18;

    let sections: [(&str, Vec<(&str, Field)>); 7] = [
        ("IDENTITY", vec![("Intl ID", Field::IntlId), ("Type", Field::ObjType), ("Country", Field::Country), ("Launched", Field::Launched)]),
        ("POSITION", vec![("Sub-point", Field::SubPoint), ("Altitude", Field::Altitude), ("Speed", Field::Speed), ("Period", Field::Period)]),
        ("FROM STATION", vec![("Az / El", Field::AzEl), ("Range", Field::Range), ("Range rate", Field::RangeRate), ("Status", Field::Status)]),
        ("ORBIT  osculating", vec![("Inclination", Field::Incl), ("RAAN", Field::Raan), ("Arg perigee", Field::ArgP), ("True anomaly", Field::TrueAnom), ("Eccentricity", Field::Ecc), ("Peri / apo", Field::PeriApo)]),
        ("RADIO", vec![("Downlink", Field::Downlink), ("Mode", Field::Mode), ("Doppler", Field::Doppler)]),
        ("ELEMENT SET", vec![("Epoch age", Field::EpochAge), ("B*", Field::Bstar), ("Mean motion", Field::MeanMotion)]),
        ("RANKING", vec![("Rank", Field::Rank), ("Pass", Field::Pass), ("Remaining", Field::Left)]),
    ];

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                width: Val::Px(INFO_BOX_W),
                flex_direction: FlexDirection::Column,
                border: UiRect::all(Val::Px(2.0)),
                ..default()
            },
            BackgroundColor(Color::Srgba(panel_bg)),
            BorderColor(accent),
            GlobalZIndex(10),     // above the ranking panel and HUD when they overlap
            Button,               // so the box reports presses: that is what makes it draggable
            Visibility::Hidden,
            InfoBox,
        ))
        .with_children(|b| {
            //Title bar: name + NORAD on the left, FOLLOW / PINNED toggle on the right
            b.spawn((
                Node {
                    padding: UiRect::axes(Val::Px(8.0), Val::Px(3.0)),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                },
                BackgroundColor(Color::Srgba(title_bg)),
            ))
            .with_children(|t| {
                t.spawn((Text::new(""), TextFont { font: font.clone(), font_size: 10.5, ..default() }, TextColor(accent), InfoTitle));
                t.spawn((
                    Button,
                    ButtonAction::FollowSat,
                    Node { padding: UiRect::axes(Val::Px(6.0), Val::Px(1.0)), border: UiRect::all(Val::Px(2.0)), flex_shrink: 0.0, ..default() },
                    BackgroundColor(hex(&cfg.colors.button)),
                    BorderColor(accent),
                ))
                .with_children(|f| {
                    f.spawn((Text::new(if cfg!(target_os = "android") { "BACK CLEARS" } else { "DRAG ME" }), TextFont { font: font.clone(), font_size: 9.0, ..default() }, TextColor(dim_c), InfoFollowLabel));
                });
            });

            for (name, rows) in sections.iter() {
                //Section header strip
                b.spawn((
                    Node {
                        padding: UiRect::axes(Val::Px(8.0), Val::Px(1.0)),
                        margin: UiRect::top(Val::Px(2.0)),
                        border: UiRect::left(Val::Px(3.0)),
                        ..default()
                    },
                    BackgroundColor(Color::Srgba(header_bg)),
                    BorderColor(accent),
                ))
                .with_children(|h| {
                    h.spawn((Text::new(*name), TextFont { font: font.clone(), font_size: 9.5, ..default() }, TextColor(text_c)));
                });
                //Label / value rows
                for (label, field) in rows {
                    b.spawn(Node {
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(0.0)),
                        column_gap: Val::Px(6.0),
                        ..default()
                    })
                    .with_children(|r| {
                        r.spawn((
                            Text::new(label.to_uppercase()),
                            TextFont { font: font.clone(), font_size: 10.0, ..default() },
                            TextColor(dim_c),
                            Node { width: Val::Px(112.0), flex_shrink: 0.0, ..default() },
                        ));
                        r.spawn((Text::new("-"), TextFont { font: font.clone(), font_size: 10.0, ..default() }, TextColor(text_c), *field));
                    });
                }
            }
            b.spawn(Node { height: Val::Px(6.0), ..default() });
        });
}


fn update_hud(
    sim: Res<Sim>,
    mode: Res<Mode>,
    cfg: Res<Config>,
    in_view: Res<InView>,
    ranked_only: Res<RankedOnly>,
    regions: Res<Regions>,
    explore: Res<Explore>,
    prop: Res<PropStatus>,
    cats: Res<Categories>, tfilter: Res<TypeFilter>,
    mut q: Query<&mut Text, With<HudText>>,
) {
    let Ok(mut text) = q.get_single_mut() else { return };
    let region_name = regions.list.get(regions.current).map_or("mask".to_string(), |r| r.name.clone());
    let mut state = match *mode {
        Mode::Live if sim.exhausted => "STOPPED  (DATA ENDED - UPDATE TLES)".to_string(),
        Mode::Live => "LIVE".to_string(),
        Mode::History => format!("HISTORY  x{:.0}  {}", sim.speed, if sim.paused { "PAUSED" } else { "RUNNING" }),
    };
    if !prop.finished && prop.total > 0 { state = format!("{state}    PROPAGATING {}/{}", prop.done, prop.total); }
    let utc = fmt_utc_est(chrono::Utc::now());
    let sim_utc = jd_to_string(sim.jd0 + sim.t / 86400.0);
    let new_text = format!(
        "PERIGEE // ORBIT VIEW\n\
         now  {utc}\n\
         sim  {sim_utc}    {state}\n\
         {}    view {region_name}    in view {}{}{}{}",
        cfg.station.name, in_view.count, if ranked_only.0 { "    RANKED ONLY" } else { "" },
        if tfilter.0.is_some() { format!("    TYPE {}", cats.label(&tfilter)) } else { String::new() },
        if explore.on { if cfg!(target_os = "android") { "    EXPLORE  (REWIND ENDS)" } else { "    EXPLORE  (X ENDS)" } } else { "" },
    ).to_uppercase();
    if text.0 != new_text { text.0 = new_text; }
}

//------------------------------------------------------------------------------------------ controls
fn toggle_mode(mode: &mut Mode, sim: &mut Sim) {
    *mode = match *mode { Mode::History => Mode::Live, Mode::Live => Mode::History };
    sim.set_mode(*mode);
}

fn keyboard(
    keys: Res<ButtonInput<KeyCode>>, cfg: Res<Config>, search: Res<Search>, time: Res<Time>, mut explore: ResMut<Explore>,
    mut sim: ResMut<Sim>, mut sel: ResMut<Selected>, mut mode: ResMut<Mode>, mut ranked_only: ResMut<RankedOnly>,
    mut panel_hidden: ResMut<PanelHidden>,
    mut regions: ResMut<Regions>,
    mut panel: Query<&mut Visibility, With<RankPanel>>,
    mut tmenu: ResMut<TypeMenu>, tfilter: Res<TypeFilter>, focus: Res<ViewerFocus>,
) {
    if search.active || !focus.0 { return; }   // typing goes to the search box, or to perigee-control's tile
    if keys.just_pressed(KeyCode::KeyK) { ranked_only.0 = !ranked_only.0; }
    if keys.just_pressed(KeyCode::KeyT) { tmenu.open = !tmenu.open; tmenu.highlight = tfilter.0.map_or(0, |k| k + 1); }
    if keys.just_pressed(KeyCode::KeyP) {
        panel_hidden.0 = !panel_hidden.0;
        for mut v in &mut panel { *v = if panel_hidden.0 { Visibility::Hidden } else { Visibility::Inherited }; }
    }
    if keys.just_pressed(KeyCode::KeyL) { toggle_mode(&mut mode, &mut sim); }
    if keys.just_pressed(KeyCode::KeyV) && !regions.list.is_empty() { regions.current = (regions.current + 1) % regions.list.len(); }
    if keys.just_pressed(KeyCode::Space) { sim.paused = !sim.paused; }
    if keys.just_pressed(KeyCode::Equal) || keys.just_pressed(KeyCode::NumpadAdd) { sim.speed = (sim.speed * 2.0).min(cfg.sim.max_speed); }
    if keys.just_pressed(KeyCode::Minus) || keys.just_pressed(KeyCode::NumpadSubtract) { sim.speed = (sim.speed / 2.0).max(cfg.sim.min_speed); }
    if keys.just_pressed(KeyCode::KeyR) { sim.t = 0.0; }
    if keys.just_pressed(KeyCode::KeyX) { let on = !explore.on; set_explore(on, time.elapsed_secs_f64(), &mut explore, &mut sel, &mut panel_hidden, &mut panel); }
    if keys.just_pressed(KeyCode::Escape) { set_explore(false, time.elapsed_secs_f64(), &mut explore, &mut sel, &mut panel_hidden, &mut panel); sel.0 = None; }
}

fn buttons(
    mut q: Query<(&Interaction, &ButtonAction, &mut BackgroundColor), (Changed<Interaction>, With<Button>)>,
    cfg: Res<Config>,
    mut sim: ResMut<Sim>,
    mut sel: ResMut<Selected>,
    mut mode: ResMut<Mode>,
    mut ranked_only: ResMut<RankedOnly>,
    mut search: ResMut<Search>,
    mut score_open: ResMut<ScoreOpen>,
    mut info_drag: ResMut<InfoDrag>,
    mut menu: ResMut<RegionMenu>,
    regions: Res<Regions>,
    mut tmenu: ResMut<TypeMenu>,
    tfilter: Res<TypeFilter>,
) {
    for (interaction, action, mut bg) in &mut q {
        match interaction {
            Interaction::Pressed => {
                *bg = BackgroundColor(hex(&cfg.colors.button_hover));
                match action {
                    ButtonAction::ToggleMode => toggle_mode(&mut mode, &mut sim),
                    ButtonAction::TogglePause => sim.paused = !sim.paused,
                    ButtonAction::Slower => sim.speed = (sim.speed / 2.0).max(cfg.sim.min_speed),
                    ButtonAction::Faster => sim.speed = (sim.speed * 2.0).min(cfg.sim.max_speed),
                    ButtonAction::Restart => sim.t = 0.0,
                    ButtonAction::Clear => sel.0 = None,
                    ButtonAction::ToggleRanked => ranked_only.0 = !ranked_only.0,
                    ButtonAction::OpenSearch => { search.active = true; }
                    ButtonAction::ToggleScore => score_open.0 = !score_open.0,
                    ButtonAction::FollowSat => { info_drag.pinned = None; info_drag.dragging = false; }
                    ButtonAction::RegionMenu => { menu.open = !menu.open; menu.highlight = regions.current; }
                    ButtonAction::TypeMenu => { tmenu.open = !tmenu.open; tmenu.highlight = tfilter.0.map_or(0, |k| k + 1); }
                }
            }
            Interaction::Hovered => *bg = BackgroundColor(hex(&cfg.colors.button_hover)),
            Interaction::None => *bg = BackgroundColor(hex(&cfg.colors.button)),
        }
    }
}

//Clicking a ranking row selects that satellite; the selected row stays tinted in the selection colour
fn rank_rows(
    mut clicks: Query<(&Interaction, &RankRow), (Changed<Interaction>, With<Button>)>,
    mut rows: Query<(&Interaction, &RankRow, &RankRowIndex, &mut BackgroundColor), With<Button>>,
    cfg: Res<Config>,
    cursor: Res<RowCursor>,
    mut sel: ResMut<Selected>,
) {
    for (interaction, row) in &mut clicks {
        if *interaction == Interaction::Pressed { sel.0 = Some(row.0); }
    }
    let mut selected_tint = hex(&cfg.colors.selected).to_srgba(); selected_tint.alpha = 0.45;
    for (interaction, row, idx, mut bg) in &mut rows {
        let want = if sel.0 == Some(row.0) { Color::Srgba(selected_tint) }
                   else if cursor.0 == Some(idx.0) || *interaction == Interaction::Hovered { hex(&cfg.colors.button_hover) }
                   else { hex(&cfg.colors.button) };
        if bg.0 != want { bg.0 = want; }
    }
}

//Drop-down option rows: click selects; highlight follows the remote cursor
fn region_options(
    mut rows: Query<(&Interaction, &RegionOption, &mut BackgroundColor), With<Button>>,
    cfg: Res<Config>,
    menu: Res<RegionMenu>,
    mut regions: ResMut<Regions>,
) {
    for (interaction, opt, mut bg) in &mut rows {
        if *interaction == Interaction::Pressed && regions.current != opt.0 { regions.current = opt.0; }
        let want = if regions.current == opt.0 { hex(&cfg.colors.button_hover) }
                   else if menu.open && menu.highlight == opt.0 || *interaction == Interaction::Hovered { hex(&cfg.colors.button_hover) }
                   else { hex(&cfg.colors.button) };
        if bg.0 != want { bg.0 = want; }
    }
}

//TYPE drop-down rows: click selects; highlight follows the remote cursor (row 0 = ALL)
fn type_options(
    mut rows: Query<(&Interaction, &TypeOption, &mut BackgroundColor), With<Button>>,
    cfg: Res<Config>,
    menu: Res<TypeMenu>,
    cats: Res<Categories>,
    mut filter: ResMut<TypeFilter>,
) {
    for (interaction, opt, mut bg) in &mut rows {
        if *interaction == Interaction::Pressed {
            let want = cats.filter_for_row(opt.0);
            if filter.0 != want { filter.0 = want; }
        }
        let current = filter.0.map_or(0, |k| k + 1) == opt.0;
        let want = if current || (menu.open && menu.highlight == opt.0) || *interaction == Interaction::Hovered { hex(&cfg.colors.button_hover) }
                   else { hex(&cfg.colors.button) };
        if bg.0 != want { bg.0 = want; }
    }
}

//TYPE header caption and list visibility
fn apply_type(
    filter: Res<TypeFilter>,
    cats: Res<Categories>,
    menu: Res<TypeMenu>,
    mut header: Query<&mut Text, With<TypeHeader>>,
    mut list: Query<&mut Visibility, With<TypeList>>,
) {
    let caption = format!("TYPE: {}  [{}]{}", cats.label(&filter), if menu.open { "-" } else { "+" },
        if cfg!(target_os = "android") { "  FAST FORWARD x2" } else { "  T" });
    for mut t in &mut header { if t.0 != caption { t.0 = caption.clone(); } }
    for mut v in &mut list { let want = if menu.open { Visibility::Inherited } else { Visibility::Hidden }; if *v != want { *v = want; } }
}

//CATEGORIES.json (written by `perigee` / `perigee categories`) -> per category, which columns belong.
//Membership is by NORAD ID, mapped onto the viewer's column order through the catalog ids.
fn load_categories(txt: Option<&str>, ids: &[Option<u32>]) -> Categories {
    let mut out = Categories { total: ids.len(), ..default() };
    let Some(txt) = txt else { return out };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(txt) else { eprintln!("CATEGORIES.json: not valid JSON"); return out };
    for c in v["categories"].as_array().into_iter().flatten() {
        let Some(name) = c["name"].as_str() else { continue };
        let set: HashSet<u32> = c["norads"].as_array().into_iter().flatten().filter_map(|n| n.as_u64().map(|n| n as u32)).collect();
        let members: Vec<bool> = ids.iter().map(|id| id.is_some_and(|id| set.contains(&id))).collect();
        out.counts.push(members.iter().filter(|m| **m).count());
        out.names.push(name.to_uppercase());
        out.members.push(members);
    }
    if !out.members.is_empty() {
        let other: Vec<bool> = (0..ids.len()).map(|c| !out.members.iter().any(|m| m[c])).collect();
        out.counts.push(other.iter().filter(|m| **m).count());
        out.names.push("OTHER".into());
        out.members.push(other);
    }
    out
}

//Header caption, list visibility, wedge visibility; and tell Perigee when the selection changed
fn apply_region(
    regions: Res<Regions>,
    menu: Res<RegionMenu>,
    source: Res<DataSource>,
    mut header: Query<&mut Text, With<RegionHeader>>,
    mut list: Query<&mut Visibility, With<RegionList>>,
    mut rr: ResMut<Rerank>,
    wanted: Res<RegionWanted>,
    mut sent: Local<Option<usize>>,
    mut started: Local<bool>,
) {
    if let Some(r) = regions.list.get(regions.current) {
        let caption = format!("VIEW: {}  [{}]{}", r.name, if menu.open { "-" } else { "+" },
            if cfg!(target_os = "android") { "  FAST FORWARD" } else { "  V" });
        for mut t in &mut header { if t.0 != caption { t.0 = caption.clone(); } }
    }
    for mut v in &mut list { let want = if menu.open { Visibility::Inherited } else { Visibility::Hidden }; if *v != want { *v = want; } }

    //Selection changed since we last told Perigee: write / post it and ask for a re-rank right away
    if !*started { *started = true; *sent = regions.sent; }
    if *sent != Some(regions.current) {
        if let Some(r) = regions.list.get(regions.current) {
            //The send goes on its own thread so a slow network never stalls the frame loop
            match serde_json::to_string(r) {
                Ok(json) => {
                    let (src, name) = (source.0.clone(), r.name.clone());
                    if let Ok(mut w) = wanted.0.lock() { *w = Some((r.name.clone(), std::time::Instant::now() + std::time::Duration::from_secs(90))); }
                    std::thread::spawn(move || match src.send_region(&json) {
                        Ok(()) => println!("view region -> Perigee: {name}"),
                        Err(e) => eprintln!("could not send the view region: {e}"),
                    });
                    rr.next = 0.0;
                }
                Err(e) => eprintln!("region json: {e}"),
            }
        }
        *sent = Some(regions.current);
    }
}

//TV remote (and desktop arrow keys): Up/Down move the row cursor, Select/Enter picks that satellite,
//Left/Right step the selection through the ranked list, Play/Pause toggles ranked-only, Back clears
//the selection (and with nothing selected, Back leaves the app the normal Android way).
fn remote_controls(
    mut events: EventReader<KeyboardInput>,
    search: Res<Search>,
    ranks: Res<Ranks>,
    mut cursor: ResMut<RowCursor>,
    mut sel: ResMut<Selected>,
    mut ranked_only: ResMut<RankedOnly>,
    mut panel_hidden: ResMut<PanelHidden>,
    mut score_open: ResMut<ScoreOpen>,
    mut menu: ResMut<RegionMenu>,
    mut regions: ResMut<Regions>,
    mut panel: Query<&mut Visibility, With<RankPanel>>,
    time: Res<Time>,
    mut explore: ResMut<Explore>,
    (mut tmenu, mut tfilter, cats): (ResMut<TypeMenu>, ResMut<TypeFilter>, Res<Categories>),
) {
    if search.active { return; }
    let shown = ranks.entries.len();
    let now = time.elapsed_secs_f64();
    for ev in events.read() {
        if !ev.state.is_pressed() { continue; }
        //FAST FORWARD steps the drop-downs: VIEW open -> TYPE open -> both closed; the D-pad works whichever is open
        if matches!(&ev.logical_key, Key::MediaFastForward | Key::MediaTrackNext) {
            if menu.open { menu.open = false; tmenu.open = true; tmenu.highlight = tfilter.0.map_or(0, |k| k + 1); }
            else if tmenu.open { tmenu.open = false; }
            else { menu.open = true; menu.highlight = regions.current; }
            continue;
        }
        if tmenu.open {
            let n = cats.rows().max(1);
            match &ev.logical_key {
                Key::ArrowDown => tmenu.highlight = (tmenu.highlight + 1) % n,
                Key::ArrowUp => tmenu.highlight = (tmenu.highlight + n - 1) % n,
                Key::Enter => { tfilter.0 = cats.filter_for_row(tmenu.highlight); tmenu.open = false; }
                Key::BrowserBack => tmenu.open = false,
                _ => {}
            }
            continue;
        }
        if menu.open {
            let n = regions.list.len().max(1);
            match &ev.logical_key {
                Key::ArrowDown => menu.highlight = (menu.highlight + 1) % n,
                Key::ArrowUp => menu.highlight = (menu.highlight + n - 1) % n,
                Key::Enter => { regions.current = menu.highlight.min(n - 1); menu.open = false; }
                Key::BrowserBack => menu.open = false,
                _ => {}
            }
            continue;
        }
        //Fire TV MENU button: winit has no name for it, so match the physical code or the raw Android keycode 82
        let is_menu = ev.key_code == KeyCode::ContextMenu
            || matches!(&ev.logical_key, Key::Unidentified(bevy::input::keyboard::NativeKey::Android(82)));
        if is_menu {
            panel_hidden.0 = !panel_hidden.0;
            for mut v in &mut panel { *v = if panel_hidden.0 { Visibility::Hidden } else { Visibility::Inherited }; }
            continue;
        }
        match &ev.logical_key {
            Key::MediaRewind | Key::MediaTrackPrevious => { let on = !explore.on; set_explore(on, now, &mut explore, &mut sel, &mut panel_hidden, &mut panel); }
            Key::ArrowDown => {
                if shown > 0 { cursor.0 = Some(cursor.0.map_or(0, |c| (c + 1) % shown)); }
            }
            Key::ArrowUp => {
                if shown > 0 { cursor.0 = Some(cursor.0.map_or(shown - 1, |c| (c + shown - 1) % shown)); }
            }
            Key::Enter => {
                //A row picks that satellite; SELECT with no row highlighted opens / closes the score weights
                match cursor.0 {
                    Some(c) => { if let Some(e) = ranks.entries.get(c) { sel.0 = Some(e.pass.column); } }
                    None => { score_open.0 = !score_open.0; }
                }
            }
            Key::ArrowRight | Key::ArrowLeft => {
                if ranks.entries.is_empty() { continue; }
                let cur = sel.0.and_then(|col| ranks.entries.iter().position(|e| e.pass.column == col));
                let n = ranks.entries.len();
                let next = match (cur, &ev.logical_key) {
                    (None, _) => 0,
                    (Some(i), Key::ArrowRight) => (i + 1) % n,
                    (Some(i), _) => (i + n - 1) % n,
                };
                sel.0 = Some(ranks.entries[next].pass.column);
                cursor.0 = if next < shown { Some(next) } else { None };
            }
            Key::MediaPlayPause => { ranked_only.0 = !ranked_only.0; }
            Key::BrowserBack => { if explore.on { set_explore(false, now, &mut explore, &mut sel, &mut panel_hidden, &mut panel); } else if sel.0.is_some() { sel.0 = None; } }
            _ => {}
        }
    }
}

//Search box: "/" opens it, type a name fragment or NORAD number, up/down to pick, Enter selects, Esc closes
fn search_input(
    mut events: EventReader<KeyboardInput>,
    primary: Query<Entity, With<PrimaryWindow>>,
    focus: Res<ViewerFocus>,
    cat: Res<Catalog>,
    orbits: Res<Orbits>,
    mut search: ResMut<Search>,
    mut sel: ResMut<Selected>,
) {
    if !focus.0 { events.clear(); return; }   // perigee-control's own tile has the keyboard
    let mut changed = false;
    for ev in events.read() {
        if primary.get(ev.window).is_err() { continue; }   // typed into another window (perigee-control's command page)
        if !ev.state.is_pressed() { continue; }
        if !search.active {
            if ev.logical_key == Key::Character("/".into()) { search.active = true; search.query.clear(); changed = true; }
            continue;
        }
        match &ev.logical_key {
            Key::Escape => { search.active = false; }
            Key::Enter => {
                if let Some(&col) = search.results.get(search.highlight) { sel.0 = Some(col); }
                search.active = false;
            }
            Key::Backspace => { search.query.pop(); changed = true; }
            Key::ArrowDown => { if search.highlight + 1 < search.results.len() { search.highlight += 1; } }
            Key::ArrowUp => { search.highlight = search.highlight.saturating_sub(1); }
            Key::Space => { search.query.push(' '); changed = true; }
            Key::Character(c) => {
                if c != "/" && search.query.len() < 32 { search.query.push_str(c); changed = true; }
            }
            _ => {}
        }
    }
    if !changed { return; }

    //Match by NORAD prefix or case-insensitive name fragment, best (shortest name) first, at most 8
    let q = search.query.trim().to_lowercase();
    let mut hits: Vec<(usize, String)> = Vec::new();
    if !q.is_empty() {
        for i in 0..orbits.0.len() {
            let id = cat.ids.get(i).copied().flatten();
            let name = id.and_then(|id| cat.names.get(&id)).cloned().unwrap_or_default();
            let id_match = id.map_or(false, |id| id.to_string().starts_with(&q));
            let name_match = !name.is_empty() && name.to_lowercase().contains(&q);
            if id_match || name_match { hits.push((i, name)); }
        }
    }
    hits.sort_by_key(|(_, n)| n.len());
    search.results = hits.into_iter().take(8).map(|(i, _)| i).collect();
    search.highlight = 0;
}

fn update_search_ui(
    search: Res<Search>,
    cat: Res<Catalog>,
    mut box_q: Query<&mut Text, (With<SearchBox>, Without<SearchResults>)>,
    mut res_q: Query<&mut Text, (With<SearchResults>, Without<SearchBox>)>,
) {
    if !search.is_changed() { return; }
    if let Ok(mut t) = box_q.get_single_mut() {
        t.0 = if search.active { format!("SEARCH  {}\u{2588}", search.query.to_uppercase()) }
              else if search.query.is_empty() { "SEARCH  /  NAME OR NORAD".to_string() }
              else { format!("SEARCH  {}", search.query.to_uppercase()) };
    }
    if let Ok(mut t) = res_q.get_single_mut() {
        if !search.active { t.0.clear(); return; }
        let mut lines = String::new();
        for (k, &col) in search.results.iter().enumerate() {
            let marker = if k == search.highlight { ">" } else { " " };
            lines.push_str(&format!("{marker} {}\n", describe(&cat, col).to_uppercase()));
        }
        if search.results.is_empty() && !search.query.is_empty() { lines.push_str("  no match"); }
        t.0 = lines;
    }
}

//Every 2 s, if SATELLITE_RANKS.json has a new modified time (or F5 was pressed), reload it and rebuild the panel
fn watch_ranks(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    cfg: Res<Config>,
    orbits: Res<Orbits>,
    mut watch: ResMut<RankWatch>,
    remote: Res<RemoteRanks>,
    mut ranks: ResMut<Ranks>,
    score_open: Res<ScoreOpen>,
    panel_hidden: Res<PanelHidden>,
    ui_font: Res<UiFont>,
    regions: Res<Regions>,
    cats: Res<Categories>,
    mut commands: Commands,
    panel: Query<Entity, With<RankPanel>>,
) {
    if keys.just_pressed(KeyCode::F5) { watch.force = true; }
    let now = time.elapsed_secs_f64();

    //Remote source: rankings come in over the channel from the poller thread
    if let Some(txt) = remote.0.lock().ok().and_then(|rx| rx.try_recv().ok()) {
        let fresh = parse_ranks(Some(&txt), orbits.0.len(), cfg.data.top_ranked);
        println!("rankings received: {} entries", fresh.entries.len());
        *ranks = fresh;
        for e in &panel { commands.entity(e).despawn_recursive(); }
        spawn_rank_panel(&mut commands, &cfg, &ranks, &regions, &cats, score_open.0, panel_hidden.0, &ui_font.0);
        return;
    }
    if watch.path.as_os_str().is_empty() { return; }

    if now < watch.next_check && !watch.force { return; }
    watch.next_check = now + cfg.data.ranks_poll_seconds.max(1.0);

    let modified = std::fs::metadata(&watch.path).and_then(|m| m.modified()).ok();
    if modified == watch.last_modified && !watch.force { return; }
    watch.last_modified = modified;
    watch.force = false;

    let txt = std::fs::read_to_string(&watch.path).ok();
    let fresh = parse_ranks(txt.as_deref(), orbits.0.len(), cfg.data.top_ranked);
    println!("rankings reloaded: {} entries", fresh.entries.len());
    *ranks = fresh;

    for e in &panel { commands.entity(e).despawn_recursive(); }
    spawn_rank_panel(&mut commands, &cfg, &ranks, &regions, &cats, score_open.0, panel_hidden.0, &ui_font.0);
}

//Fill the score section: weights + settings, and weight x term = contribution for the selected satellite
fn update_score_detail(
    sel: Res<Selected>,
    ranks: Res<Ranks>,
    score_open: Res<ScoreOpen>,
    cat: Res<Catalog>,
    mut detail: Query<(&mut Text, &mut Visibility), With<ScoreDetail>>,
    mut labels: Query<(&Parent, &mut Text), (Without<ScoreDetail>, With<Parent>)>,
    buttons: Query<(Entity, &ButtonAction), With<Button>>,
) {
    if !(sel.is_changed() || ranks.is_changed() || score_open.is_changed()) { return; }

    //Button caption follows the open / closed state
    for (entity, action) in &buttons {
        if let ButtonAction::ToggleScore = action {
            for (parent, mut t) in &mut labels {
                if parent.get() == entity {
                    let suffix = if cfg!(target_os = "android") { "  REWIND" } else { "" };
                    t.0 = format!("SCORE WEIGHTS  [{}]{}", if score_open.0 { "-" } else { "+" }, suffix);
                }
            }
        }
    }

    let Ok((mut text, mut vis)) = detail.get_single_mut() else { return };
    *vis = if score_open.0 { Visibility::Inherited } else { Visibility::Hidden };
    if !score_open.0 { return; }

    let w = ranks.weights.clone().unwrap_or(Weights { duration: 0.35, elevation: 0.30, transmitter: 0.25, freshness: 0.10 });
    let mut out = String::new();
    if ranks.weights.is_none() {
        out.push_str("(weights not in this SATELLITE_RANKS.json, showing defaults; re-run perigee rank)\n");
    }
    if !ranks.generated_local.is_empty() {
        out.push_str(&format!("ranking of passes above {:.0} deg starting within {:.0} min, made {}\n\n",
            ranks.mask_deg, ranks.horizon_min, ranks.generated_local.get(11..19).unwrap_or("")));
    }
    //What the code favours: one row per parameter, weight = its share of a perfect 1.00 score
    out.push_str("WHAT IS FAVOURED             weight   max boost\n");
    out.push_str(&format!("time left above mask          {:.2}     +{:.2}\n", w.duration, w.duration));
    out.push_str("    full at 10 min or more, scales down, 0 under 2 min\n");
    out.push_str(&format!("best elevation still ahead    {:.2}     +{:.2}\n", w.elevation, w.elevation));
    out.push_str("    sin(el): 30 deg = half, 90 = full; above 85 x0.3 (keyhole)\n");
    out.push_str(&format!("transmitter known             {:.2}     +{:.2}\n", w.transmitter, w.transmitter));
    out.push_str("    downlink freq listed = full, listed w/o freq = half, none = 0\n");
    out.push_str(&format!("element set freshness         {:.2}     +{:.2}\n", w.freshness, w.freshness));
    out.push_str("    full when brand new, 0 at 24 h old, linear between\n");
    let total = w.duration + w.elevation + w.transmitter + w.freshness;
    out.push_str(&format!("perfect score                 {:.2}     +{:.2}\n", total, total));
    out.push_str("not scored: range, azimuth, band, in progress vs upcoming\n");

    match sel.0 {
        None => out.push_str("\nselect a satellite to see its breakdown"),
        Some(col) => match ranks.entries.iter().find(|e| e.pass.column == col) {
            None => out.push_str(&format!("\n{}  is not on the ranking list", describe(&cat, col))),
            Some(e) => {
                out.push_str(&format!("\n#{}  {}   weight x term = boost\n", e.rank, e.name));
                out.push_str(&format!("  duration     {:.2} x {:.2} = {:.3}\n", w.duration, e.duration_term, w.duration * e.duration_term));
                out.push_str(&format!("  elevation    {:.2} x {:.2} = {:.3}\n", w.elevation, e.elevation_term, w.elevation * e.elevation_term));
                out.push_str(&format!("  transmitter  {:.2} x {:.2} = {:.3}\n", w.transmitter, e.transmitter_term, w.transmitter * e.transmitter_term));
                out.push_str(&format!("  freshness    {:.2} x {:.2} = {:.3}\n", w.freshness, e.freshness_term, w.freshness * e.freshness_term));
                out.push_str(&format!("  score {:.3}", e.score));
            }
        },
    }
    text.0 = out;
}

//Press anywhere on the info box and move the mouse to drag it; it stays where it is dropped
fn drag_info_box(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<&Camera, With<Camera3d>>,
    boxq: Query<&Interaction, With<InfoBox>>,
    follow_btn: Query<&Interaction, (With<Button>, Without<InfoBox>)>,
    ui_scale: Res<UiScale>,
    mut drag: ResMut<InfoDrag>,
) {
    let Ok(window) = windows.get_single() else { return };
    let Ok(camera) = cam_q.get_single() else { return };
    let Some(cursor) = viewport_cursor(window, camera) else { drag.dragging = false; return };
    let cursor = cursor / ui_scale.0.max(0.01);
    let over_box = boxq.get_single().map_or(false, |i| *i != Interaction::None);
    let over_other_button = follow_btn.iter().any(|i| *i == Interaction::Pressed);

    if buttons.just_pressed(MouseButton::Left) && over_box && !over_other_button {
        drag.dragging = true;
        drag.grab_offset = cursor - drag.pos;
    }
    if drag.dragging {
        if buttons.pressed(MouseButton::Left) {
            drag.pinned = Some(cursor - drag.grab_offset);
        } else {
            drag.dragging = false;
        }
    }
}

//Fill the info box beside the selected satellite's marker, every frame
fn update_info_box(
    sel: Res<Selected>,
    sim: Res<Sim>,
    cat: Res<Catalog>,
    cfg: Res<Config>,
    orbits: Res<Orbits>,
    ranks: Res<Ranks>,
    regions: Res<Regions>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    sats: Query<(&Satellite, &GlobalTransform)>,
    mut boxq: Query<(&mut Node, &mut Visibility, &ComputedNode), With<InfoBox>>,
    extra: (Res<PanelHidden>, Local<Option<usize>>, Res<UiScale>,
            Query<(&ComputedNode, &GlobalTransform), (With<RankPanel>, Without<InfoBox>)>,
            Query<(&ComputedNode, &GlobalTransform), (With<HudText>, Without<InfoBox>)>),   // bundled: Bevy systems take at most 16 parameters
    mut title: Query<&mut Text, (With<InfoTitle>, Without<Field>, Without<InfoFollowLabel>)>,
    mut follow_label: Query<(&mut Text, &mut TextColor), (With<InfoFollowLabel>, Without<Field>, Without<InfoTitle>)>,
    mut fields: Query<(&Field, &mut Text, &mut TextColor), (Without<InfoTitle>, Without<InfoFollowLabel>)>,
    mut drag: ResMut<InfoDrag>,
) {
    let (panel_hidden, mut slot, ui_scale, panel_q, hud_q) = extra;
    //Screen rectangle of a laid-out UI node, in logical UI pixels
    let ui_rect = |cn: &ComputedNode, gt: &GlobalTransform| -> Rect {
        let k = cn.inverse_scale_factor();
        Rect::from_center_size(gt.translation().truncate() * k, cn.size() * k)
    };
    let Ok((mut node, mut vis, computed)) = boxq.get_single_mut() else { return };
    if sel.is_changed() { drag.pinned = None; drag.dragging = false; *slot = None; }   // a new target: place it afresh
    let Some(col) = sel.0 else { *vis = Visibility::Hidden; return };
    let Ok((camera, cam_tf)) = cam_q.get_single() else { return };
    let Some((_, tf)) = sats.iter().find(|(s, _)| s.0 == col) else { *vis = Visibility::Hidden; return };

    let us = ui_scale.0.max(0.01);
    let (win_w, win_h) = camera.logical_viewport_size().map(|v| (v.x / us, v.y / us))
        .or_else(|| windows.get_single().ok().map(|w| (w.width() / us, w.height() / us))).unwrap_or((1920.0, 1080.0));
    let pos = match drag.pinned {
        //Dropped somewhere by the user: stay there (kept on screen)
        Some(p) => Vec2::new(p.x.clamp(0.0, (win_w - INFO_BOX_W).max(0.0)), p.y.clamp(0.0, (win_h - 40.0).max(0.0))),
        //Automatic: the spot that stays on screen and covers the least of the globe, the panel and the target
        None => {
            let Ok(p) = camera.world_to_viewport(cam_tf, tf.translation()) else { *vis = Visibility::Hidden; return };
            let p = p / us;
            //Real box size once laid out (ComputedNode is in physical px), else the rough constants
            let size = computed.size() * computed.inverse_scale_factor();
            let (bw, bh) = if size.x > 1.0 && size.y > 1.0 { (size.x, size.y) } else { (INFO_BOX_W, INFO_BOX_H) };
            //Globe on screen: centre and radius from the camera (Earth frame sits at the origin)
            let globe = camera.world_to_viewport(cam_tf, Vec3::ZERO).ok().map(|c| {
                let r_world = (cfg.scene.earth_radius_km / cfg.scene.km_per_unit) as f32;
                let edge = cam_tf.right() * r_world;
                let r_px = camera.world_to_viewport(cam_tf, edge).map_or(300.0, |e| (e - c).length());
                (c / us, r_px / us)
            });
            //Obstacles: the ranking panel as actually laid out (while shown), the HUD text block, the target marker
            let panel = if panel_hidden.0 { None } else { panel_q.get_single().ok().map(|(cn, gt)| ui_rect(cn, gt)) };
            let hud = hud_q.get_single().ok().map(|(cn, gt)| ui_rect(cn, gt));
            let target = Rect::from_center_size(p, Vec2::splat(90.0));
            let overlap = |a: Rect, b: Rect| { let i = a.intersect(b); if i.is_empty() { 0.0 } else { i.width() * i.height() } };
            let area = (bw * bh).max(1.0);
            let clamp = |x: f32, y: f32| Vec2::new(x.clamp(8.0, (win_w - bw - 8.0).max(8.0)), y.clamp(8.0, (win_h - bh - 8.0).max(8.0)));
            //The map, for placement purposes, is the globe plus the halo of orbit tracks around it
            let halo = globe.map(|(c, r)| Rect::from_center_size(c, Vec2::splat(2.0 * r * 1.28)));
            let gap_x = halo.map_or(8.0, |h| h.max.x + 8.0);
            let under_panel = panel.map_or(Vec2::new(win_w - bw - 8.0, 8.0), |pr| Vec2::new(pr.max.x - bw, pr.max.y + 8.0));
            let candidates = [
                clamp(under_panel.x, under_panel.y),           // 0 below the ranking panel, right-aligned with it
                clamp(gap_x, 8.0),                             // 1 gap between the map and the panel, top
                clamp(gap_x, win_h - bh - 8.0),                // 2 gap, bottom
                clamp(8.0, 8.0), clamp(8.0, win_h - bh - 8.0), // 3,4 left column
                clamp(win_w - bw - 8.0, 8.0), clamp(win_w - bw - 8.0, win_h - bh - 8.0), // 5,6 right column
                clamp(p.x + 26.0, p.y - 24.0),                 // 7 right of the marker
                clamp(p.x - 26.0 - bw, p.y - 24.0),            // 8 left of it
                clamp(p.x - bw / 2.0, p.y - 26.0 - bh),        // 9 above
                clamp(p.x - bw / 2.0, p.y + 26.0),             // 10 below
            ];
            //Weights: the map and the panel matter most, then the HUD, the target and staying near it
            let score = |pos: Vec2| {
                let b = Rect::from_corners(pos, pos + Vec2::new(bw, bh));
                let mut sc = 0.0;
                if let Some((c, r)) = globe { sc += 6.0 * overlap(b, Rect::from_center_size(c, Vec2::splat(2.0 * r))) / area; }
                if let Some(h) = halo { sc += 2.0 * overlap(b, h) / area; }
                if let Some(pr) = panel { sc += 5.0 * overlap(b, pr) / area; }
                if let Some(h) = hud { sc += 2.0 * overlap(b, h) / area; }
                sc += 4.0 * overlap(b, target) / target.width().powi(2);
                sc += 0.15 * (b.center() - p).length() / Vec2::new(win_w, win_h).length();
                sc
            };
            let scores: Vec<f32> = candidates.iter().map(|&c| score(c)).collect();
            let best = (0..candidates.len()).min_by(|&a, &b| scores[a].total_cmp(&scores[b])).unwrap_or(0);
            //Stick with the current spot unless another is clearly better, so the box does not hop around
            let keep = slot.filter(|&k| scores[k] <= scores[best] + 0.15);
            let chosen = keep.unwrap_or(best);
            *slot = Some(chosen);
            candidates[chosen]
        }
    };
    drag.pos = pos;
    node.left = Val::Px(pos.x);
    node.top = Val::Px(pos.y);
    *vis = Visibility::Inherited;
    if let Ok((mut t, mut c)) = follow_label.get_single_mut() {
        let (label, color) = if cfg!(target_os = "android") { ("BACK CLEARS  LEFT/RIGHT NEXT", hex(&cfg.colors.text_dim)) }
                             else if drag.pinned.is_some() { ("PINNED  CLICK TO FOLLOW", hex(&cfg.colors.reticle_sel)) } else { ("FOLLOWING  DRAG TO PIN", hex(&cfg.colors.text_dim)) };
        if t.0 != label { t.0 = label.into(); c.0 = color; }
    }

    let m = &orbits.0[col];
    let t = sat_t(&sim, &cat, col);
    let jd = sim.jd0 + sim.t / 86400.0;
    let id = cat.ids.get(col).copied().flatten();
    let name = id.and_then(|i| cat.names.get(&i).cloned()).unwrap_or_else(|| format!("TRACK #{col}"));
    let g = |key: &str| -> String {
        id.and_then(|i| cat.omm.get(&i)).and_then(|o| o[key].as_str()).unwrap_or("-").to_string()
    };
    if let Ok(mut tt) = title.get_single_mut() {
        tt.0 = format!("{}     NORAD {}", name.to_uppercase(), id.map_or("-".into(), |i| i.to_string()));
    }

    let text_c = hex(&cfg.colors.text);
    let good_c = hex(&cfg.colors.marker_in_view);
    let mut vals: HashMap<Field, (String, Color)> = HashMap::new();
    let mut set = |f: Field, v: String| { vals.insert(f, (v, text_c)); };

    set(Field::IntlId, g("OBJECT_ID"));
    set(Field::ObjType, g("OBJECT_TYPE"));
    set(Field::Country, g("COUNTRY_CODE"));
    set(Field::Launched, g("LAUNCH_DATE"));
    set(Field::Site, g("SITE"));

    let mut status_in_view = false;
    let mut range_rate_val = 0.0;
    if t >= 0.0 {
        let r = sat_eci_km(&cfg, m, t);
        let v = sat_eci_vel(&cfg, m, t);
        let speed = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        let gmst = gmst_rad(jd);
        let ecef = eci_to_ecef(r, gmst);
        let (lat, lon, alt) = ecef_to_geodetic(ecef);
        let st = &cfg.station;
        let sta = geodetic_to_ecef(st.lat_deg, st.lon_deg, st.alt_m);
        let (az, el, range) = look_angles(sta, st.lat_deg, st.lon_deg, ecef);
        let vecef = eci_to_ecef(v, gmst);
        let omega = 7.2921159e-5;
        let vrel = [vecef[0] + omega * ecef[1], vecef[1] - omega * ecef[0], vecef[2]];
        let los = [ecef[0]-sta[0], ecef[1]-sta[1], ecef[2]-sta[2]];
        let range_rate = (vrel[0]*los[0] + vrel[1]*los[1] + vrel[2]*los[2]) / range;
        range_rate_val = range_rate;
        let (a, e, i, raan, argp, nu) = rv_to_coe(r, v);
        let period_min = 2.0 * std::f64::consts::PI * (a * a * a / 398600.4418).sqrt() / 60.0;
        status_in_view = regions.list.get(regions.current).map_or(el >= st.elevation_mask_deg, |r| r.contains(az, el));

        set(Field::SubPoint, fmt_latlon(lat, lon));
        set(Field::Altitude, format!("{:.1} km", alt));
        set(Field::Speed, format!("{:.3} km/s", speed));
        set(Field::Period, format!("{:.1} min", period_min));
        set(Field::AzEl, format!("{:.1}  /  {:.1}", az, el));
        set(Field::Range, format!("{:.0} km", range));
        set(Field::RangeRate, format!("{:+.3} km/s   {}", range_rate, if range_rate < 0.0 { "approaching" } else { "receding" }));
        set(Field::Status, if status_in_view { "IN VIEW".into() } else if el > 0.0 { format!("outside {}", regions.list.get(regions.current).map_or("mask".to_string(), |r| r.name.clone())) } else { "below horizon".into() });
        set(Field::Incl, format!("{:.3} deg", i));
        set(Field::Raan, format!("{:.3} deg", raan));
        set(Field::ArgP, format!("{:.2} deg", argp));
        set(Field::TrueAnom, format!("{:.2} deg", nu));
        set(Field::Ecc, format!("{:.5}", e));
        set(Field::PeriApo, format!("{:.0}  /  {:.0} km", a * (1.0 - e) - 6378.137, a * (1.0 + e) - 6378.137));
    } else {
        for f in [Field::SubPoint, Field::Altitude, Field::Speed, Field::Period, Field::AzEl, Field::Range, Field::RangeRate,
                  Field::Incl, Field::Raan, Field::ArgP, Field::TrueAnom, Field::Ecc, Field::PeriApo] {
            set(f, "-".into());
        }
        set(Field::Status, "before element set epoch".into());
    }

    match id.and_then(|i| cat.transmitters.get(&i)) {
        Some(txs) => {
            let first = txs.iter().find(|tx| tx["downlink_low"].as_f64().is_some());
            set(Field::TxCount, format!("{}", txs.len()));
            match first {
                Some(tx) => {
                    let f = tx["downlink_low"].as_f64().unwrap_or(0.0);
                    set(Field::Downlink, format!("{:.3} MHz   {}", f / 1e6, tx["description"].as_str().unwrap_or("")));
                    set(Field::Mode, format!("{}{}", tx["mode"].as_str().unwrap_or("-"),
                        tx["baud"].as_f64().map_or(String::new(), |b| format!("   {:.0} bd", b))));
                    set(Field::Doppler, if t >= 0.0 { format!("{:+.2} kHz", -range_rate_val / 299792.458 * f / 1e3) } else { "-".into() });
                }
                None => { set(Field::Downlink, "no frequency listed".into()); set(Field::Mode, "-".into()); set(Field::Doppler, "-".into()); }
            }
        }
        None => { set(Field::TxCount, "none listed".into()); set(Field::Downlink, "-".into()); set(Field::Mode, "-".into()); set(Field::Doppler, "-".into()); }
    }

    match &cat.elsets {
        Some(es) if col < es.ncols() => {
            set(Field::EpochAge, format!("{:.1} h   ({})", (jd - es[(1, col)]) * 24.0, jd_to_string(es[(1, col)]).get(0..19).unwrap_or("")));
            set(Field::Bstar, format!("{:.3e}", es[(8, col)]));
            set(Field::MeanMotion, format!("{:.5} rev/day", es[(2, col)]));
        }
        _ => { set(Field::EpochAge, "-".into()); set(Field::Bstar, "-".into()); set(Field::MeanMotion, "-".into()); }
    }

    match ranks.entries.iter().find(|e| e.pass.column == col) {
        Some(e) => {
            set(Field::Rank, format!("#{}   score {:.3}", e.rank, e.score));
            set(Field::Pass, format!("{} to {}   peak {:.0} deg", e.aos_local.get(11..16).unwrap_or("-"), e.los_local.get(11..16).unwrap_or("-"), e.max_el_deg));
            set(Field::Left, if e.in_progress { format!("{:.1} min left", e.minutes_left) } else { format!("starts in {:.0} min", e.minutes_until_aos) });
        }
        None => { set(Field::Rank, "not on the current list".into()); set(Field::Pass, "-".into()); set(Field::Left, "-".into()); }
    }

    //Status gets the in-view green
    if let Some(v) = vals.get_mut(&Field::Status) { if status_in_view { v.1 = good_c; } }

    for (field, mut text, mut color) in &mut fields {
        if let Some((v, c)) = vals.get(field) {
            let v = v.to_uppercase();
            if text.0 != v { text.0 = v; }
            if color.0 != *c { color.0 = *c; }
        }
    }
}

//Spawn "perigee rank" in the data folder every rerank_seconds (non-blocking); the file watcher picks up the result
fn auto_rerank(time: Res<Time>, cfg: Res<Config>, mut rr: ResMut<Rerank>, mut watch: ResMut<RankWatch>) {
    if cfg.data.rerank_command.is_empty() || cfg.data.rerank_seconds <= 0.0 { return; }
    //Reap a finished run; the new file is read on the next frame instead of waiting for the poll
    if let Some(child) = rr.child.as_mut() {
        match child.try_wait() {
            Ok(Some(status)) => { println!("perigee rank finished: {status}"); rr.child = None; watch.force = true; }
            Ok(None) => return,      // still running
            Err(e) => { eprintln!("perigee rank: {e}"); rr.child = None; }
        }
    }
    let now = time.elapsed_secs_f64();
    if now < rr.next { return; }
    rr.next = now + cfg.data.rerank_seconds.max(10.0);
    let exe = std::path::Path::new(&cfg.data.rerank_command);
    let exe = if exe.is_absolute() { exe.to_path_buf() } else { std::env::current_dir().unwrap_or_default().join(exe) };
    match std::process::Command::new(&exe).arg("rank").current_dir(&rr.dir)
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn()
    {
        Ok(child) => { println!("perigee rank started ({})", exe.display()); rr.child = Some(child); }
        Err(e) => { eprintln!("could not run {}: {e}  (set rerank_command in viewer.toml)", exe.display()); rr.next = now + 600.0; }
    }
}

//Every 2 s between re-ranks: re-time every entry from the clock, drop passes that have ended, re-sort by a live score
fn refresh_ranks_live(
    time: Res<Time>,
    sim: Res<Sim>,
    cfg: Res<Config>,
    score_open: Res<ScoreOpen>,
    panel_hidden: Res<PanelHidden>,
    ui_font: Res<UiFont>,
    regions: Res<Regions>,
    cats: Res<Categories>,
    mut ranks: ResMut<Ranks>,
    mut last: Local<f64>,
    mut commands: Commands,
    panel: Query<Entity, With<RankPanel>>,
    rows: Query<(&RankRowIndex, &Children)>,
    mut texts: Query<&mut Text>,
) {
    let now = time.elapsed_secs_f64();
    if now - *last < 2.0 { return; }
    *last = now;
    if ranks.entries.is_empty() || ranks.entries[0].pass.los_jd == 0.0 { return; }   // old file layout without pass times

    let jd = sim.jd0 + sim.t / 86400.0;
    let w = ranks.weights.clone().unwrap_or(Weights { duration: 0.35, elevation: 0.30, transmitter: 0.25, freshness: 0.10 });
    let mut changed = false;
    let mut entries = std::mem::take(&mut ranks.entries);
    let before = entries.len();
    entries.retain(|e| e.pass.los_jd > jd);
    if entries.len() != before { changed = true; }
    for e in entries.iter_mut() {
        let in_progress = e.pass.aos_jd <= jd;
        let minutes_left = (e.pass.los_jd - e.pass.aos_jd.max(jd)) * 1440.0;
        let minutes_until = (e.pass.aos_jd - jd) * 1440.0;
        //Same terms as Perigee's score.rs, re-evaluated for now. Elevation: the peak if still ahead,
        //otherwise it can only get worse, so fall back to the pass's remaining fraction of the peak.
        let d = if minutes_left < 2.0 { 0.0 } else { (minutes_left / 10.0).min(1.0) };
        let best_el = if jd < e.pass.max_el_jd { e.max_el_deg } else {
            let total = (e.pass.los_jd - e.pass.max_el_jd) * 1440.0;
            if total > 0.0 { e.max_el_deg * (minutes_left / total).clamp(0.0, 1.0) } else { 0.0 }
        };
        let el_term = { let x = best_el.to_radians().sin(); if best_el > 85.0 { x * 0.3 } else { x } };
        let score = w.duration * d + w.elevation * el_term + w.transmitter * e.transmitter_term + w.freshness * e.freshness_term;
        if (score - e.score).abs() > 1e-6 || e.in_progress != in_progress { changed = true; }
        e.score = score; e.duration_term = d; e.elevation_term = el_term;
        e.in_progress = in_progress; e.minutes_left = minutes_left; e.minutes_until_aos = minutes_until;
    }
    let order_before: Vec<usize> = ranks.columns_ordered.clone();
    entries.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    for (i, e) in entries.iter_mut().enumerate() { if e.rank != i + 1 { e.rank = i + 1; } }
    let order_now: Vec<usize> = entries.iter().map(|e| e.pass.column).collect();
    ranks.columns = order_now.iter().copied().collect();
    ranks.columns_ordered = order_now.clone();
    ranks.entries = entries;
    let _ = changed;

    //Only a changed order (or a dropped pass) needs the panel rebuilt; otherwise the rows are edited in place,
    //which avoids a one-frame gap that reads as a blink on a slow display
    if order_now != order_before {
        for ent in &panel { commands.entity(ent).despawn_recursive(); }
        spawn_rank_panel(&mut commands, &cfg, &ranks, &regions, &cats, score_open.0, panel_hidden.0, &ui_font.0);
    } else {
        for (idx, children) in &rows {
            if let Some(e) = ranks.entries.get(idx.0) {
                for child in children.iter() {
                    if let Ok(mut t) = texts.get_mut(*child) {
                        let want = rank_row_label(e);
                        if t.0 != want { t.0 = want; }
                    }
                }
            }
        }
    }
}

//Local propagation, one slice per frame. Integrates for at most propagation_budget_ms, then publishes every
//satellite that has reached its target: its 6 x M matrix replaces the placeholder in Orbits and its catalog
//epoch becomes the rebased epoch of the first stored column, which un-hides the marker. The data end
//(Sim.jd_end) follows the shortest published track. When the shortest track has less than
//propagate_min_ahead_h left, everyone is aimed propagate_ahead_h past now again, so Live never runs dry.
fn propagate_tick(mut prop: ResMut<Prop>, cfg: Res<Config>, mut orbits: ResMut<Orbits>, mut cat: ResMut<Catalog>,
                  mut sim: ResMut<Sim>, mut status: ResMut<PropStatus>) {
    let Some(p) = prop.0.as_mut() else { return };
    if status.started.is_none() { status.started = Some(std::time::Instant::now()); }

    if p.pending() == 0 {
        //Everyone is at the target: is it time to push the target out again?
        let now = now_jd();
        if let Some(end) = p.min_end_jd() {
            if end < now + cfg.data.propagate_min_ahead_h / 24.0 {
                p.set_target_jd(now + cfg.data.propagate_ahead_h / 24.0);
                status.finished = false; status.started = Some(std::time::Instant::now());
                println!("propagation: extending every track to {}", jd_to_string(now + cfg.data.propagate_ahead_h / 24.0));
            }
        }
        if p.pending() == 0 { return; }
    }

    p.run(std::time::Duration::from_secs_f64(cfg.data.propagation_budget_ms.max(1.0) / 1000.0), cfg.data.propagation_threads);
    for (i, epoch, m) in p.take_ready() {
        if let Some(slot) = orbits.0.get_mut(i) { *slot = m; }
        if let Some(e) = cat.epochs.get_mut(i) { *e = Some(epoch); }
    }
    if let Some(end) = p.min_end_jd() {
        sim.jd_end = end;
        sim.t_max = ((end - sim.jd0) * 86400.0).max(0.0);
    }
    status.total = p.len();
    status.done = p.len() - p.pending();
    if p.pending() == 0 && !status.finished {
        status.finished = true;
        let secs = status.started.map_or(0.0, |t| t.elapsed().as_secs_f64());
        println!("propagation: {} tracks ready in {:.1} s, data good to {}", p.len(), secs, jd_to_string(sim.jd_end));
    }
}

fn advance_time(time: Res<Time>, mode: Res<Mode>, mut sim: ResMut<Sim>) {
    match *mode {
        Mode::Live => {
            //Locked to the wall clock; never loops. At the data's end the sim stops on the last vector and
            //the UPDATE TLES notice comes up (nothing to propagate from until Perigee runs again).
            let t = ((now_jd() - sim.jd0) * 86400.0).max(0.0);
            sim.exhausted = sim.t_max > 0.0 && t >= sim.t_max;
            sim.t = if sim.exhausted { sim.t_max } else { t };
        }
        Mode::History => {
            if sim.paused { return; }
            sim.t += time.delta_secs_f64() * sim.speed;
            if sim.t > sim.t_max { sim.t = 0.0; }
        }
    }
}

//Click (press + release without dragging) picks the nearest marker on screen.
/// The cursor in the camera's viewport (logical px from the viewport's top-left corner), None when it is
/// outside it. A camera with no viewport (the viewer on its own) gets the plain window cursor, as before;
/// inside perigee-control the globe is one tile of the window and only that tile counts.
fn viewport_cursor(window: &Window, camera: &Camera) -> Option<Vec2> {
    let c = window.cursor_position()?;
    match camera.logical_viewport_rect() {
        Some(r) => if r.contains(c) { Some(c - r.min) } else { None },
        None => Some(c),
    }
}

fn pick_satellite(
    buttons: Res<ButtonInput<MouseButton>>,
    cfg: Res<Config>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    sats: Query<(&Satellite, &GlobalTransform, &Visibility)>,
    ui: Query<&Interaction, With<Button>>,
    mut drag: ResMut<DragState>,
    mut motion: EventReader<MouseMotion>,
    mut sel: ResMut<Selected>,
) {
    let Ok(window) = windows.get_single() else { return };
    let over_ui = ui.iter().any(|i| *i != Interaction::None);

    if buttons.just_pressed(MouseButton::Left) {
        drag.press_pos = window.cursor_position();
        drag.moved = 0.0;
    }
    if buttons.pressed(MouseButton::Left) {
        for ev in motion.read() { drag.moved += ev.delta.length(); }
    }
    if buttons.just_released(MouseButton::Left) {
        let was_click = drag.press_pos.is_some() && drag.moved < 4.0;
        drag.press_pos = None;
        if !was_click || over_ui { return; }
        let Ok((camera, cam_tf)) = cam_q.get_single() else { return };
        let Some(cursor) = viewport_cursor(window, camera) else { return };

        let mut best: Option<(usize, f32)> = None;
        for (sat, tf, vis) in &sats {
            if *vis == Visibility::Hidden { continue; }
            let Ok(screen) = camera.world_to_viewport(cam_tf, tf.translation()) else { continue };
            let d = screen.distance(cursor);
            if d < cfg.tracks.pick_radius_px && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((sat.0, d));
            }
        }
        sel.0 = best.map(|(i, _)| i);
    }
}

//------------------------------------------------------------------------------------------ motion
//A satellite has no data before its own epoch (History mode), so it stays hidden until then.
//Also tests each one against the station's elevation mask and recolours the ones in view.
fn move_satellites(
    sim: Res<Sim>, cat: Res<Catalog>, cfg: Res<Config>, orbits: Res<Orbits>,
    ranks: Res<Ranks>, ranked_only: Res<RankedOnly>, sel: Res<Selected>, regions: Res<Regions>,
    mats: Res<MarkerMats>, mut in_view: ResMut<InView>,
    mut q: Query<(&Satellite, &mut Transform, &mut Visibility, &mut MeshMaterial3d<StandardMaterial>, &mut SatInView)>,
    cats: Res<Categories>, tfilter: Res<TypeFilter>,
) {
    let st = &cfg.station;
    let gmst = gmst_rad(sim.jd0 + sim.t / 86400.0);
    let sta = geodetic_to_ecef(st.lat_deg, st.lon_deg, st.alt_m);
    let mut count = 0;
    for (sat, mut tf, mut vis, mut mat, mut flag) in &mut q {
        let t = sat_t(&sim, &cat, sat.0);
        //Ranked-only and the TYPE filter both hide markers; the pick always shows
        let filtered_out = (ranked_only.0 && !ranks.columns.contains(&sat.0) || !cats.allows(&tfilter, sat.0)) && sel.0 != Some(sat.0);
        *vis = if t < 0.0 || filtered_out { Visibility::Hidden } else { Visibility::Inherited };
        let eci = sat_eci_km(&cfg, &orbits.0[sat.0], t);
        tf.translation = to_scene(&cfg, eci[0], eci[1], eci[2]);

        let (_, el, _) = look_angles(sta, st.lat_deg, st.lon_deg, eci_to_ecef(eci, gmst));
        let (az, _, _) = look_angles(sta, st.lat_deg, st.lon_deg, eci_to_ecef(eci, gmst));
        let visible = t >= 0.0 && !filtered_out && regions.list.get(regions.current).map_or(el >= st.elevation_mask_deg, |r| r.contains(az, el));
        if flag.0 != visible { flag.0 = visible; }
        if visible { count += 1; }
        let want = if sel.0 == Some(sat.0) { &mats.selected } else if visible { &mats.in_view } else { &mats.normal };
        if mat.0 != *want { mat.0 = want.clone(); }
    }
    in_view.count = count;
}

fn spin_markers(time: Res<Time>, cfg: Res<Config>, sel: Res<Selected>, mut q: Query<(&Satellite, &mut Transform)>) {
    let a = time.elapsed_secs() * cfg.scene.marker_spin;
    let cross = cfg.scene.marker_style.eq_ignore_ascii_case("cross");
    for (sat, mut tf) in &mut q {
        let s = if cross { 0.0 } else if sel.0 == Some(sat.0) { cfg.scene.selected_scale } else { 1.0 };
        tf.rotation = Quat::from_rotation_y(a) * Quat::from_rotation_x(0.9);
        tf.scale = Vec3::splat(s);
    }
}

//Trajectories are inertial; the Earth frame turns by Greenwich sidereal time so the texture,
//the station dot and the satellites all agree on where the planet is pointing.
fn spin_earth(sim: Res<Sim>, mut q: Query<&mut Transform, With<Earth>>) {
    let gmst = gmst_rad(sim.jd0 + sim.t / 86400.0);
    for mut tf in &mut q {
        tf.rotation = Quat::from_rotation_y(gmst as f32);
    }
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    cfg: Res<Config>,
    drag: Res<InfoDrag>,
    time: Res<Time>,
    panel_hidden: Res<PanelHidden>,
    windows: Query<&Window, With<PrimaryWindow>>,
    station: Query<&GlobalTransform, (With<StationDot>, Without<OrbitCamera>)>,
    mut motion: EventReader<MouseMotion>,
    mut wheel: EventReader<MouseWheel>,
    primary: Query<Entity, With<PrimaryWindow>>,
    mut q: Query<(&mut OrbitCamera, &mut Transform, &Camera), Without<StationDot>>,
    mut offset: Local<Option<f32>>,
    fly: (Res<Selected>, Query<(&Satellite, &GlobalTransform), Without<OrbitCamera>>, Local<CamFly>, Res<Explore>),
) {
    let (sel, sats, mut fly, explore) = fly;
    let fly_seconds = if explore.on { cfg.camera.fly_seconds * 1.8 } else { cfg.camera.fly_seconds };
    let c = &cfg.camera;
    //Panel hidden: the globe glides to the centre of the window; shown: back to globe_offset_x
    let target = if panel_hidden.0 { 0.0 } else { c.globe_offset_x };
    let cur = offset.get_or_insert(target);
    *cur += (target - *cur) * (1.0 - (-6.0 * time.delta_secs()).exp());
    let offset_x = *cur;
    let Ok((mut cam, mut tf, camera)) = q.get_single_mut() else { return };
    let aspect = camera.logical_viewport_size().map(|v| v.x / v.y.max(1.0)).unwrap_or(1.78);
    //Scroll zoom only counts with the cursor over the globe (its tile inside perigee-control)
    let wheel_here = windows.get_single().ok().map_or(true, |w| viewport_cursor(w, camera).is_some());
    let dragging = buttons.pressed(MouseButton::Left) && !drag.dragging;
    //The camera transform for a given orbit state (pivot, angles, distance from the pivot), exactly as it is
    //applied at the end of this system
    let place = |pivot: Vec3, yaw: f32, pitch: f32, dist: f32| -> Transform {
        let rot = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(-pitch);
        let mut t = Transform::from_translation(pivot + rot * Vec3::new(0.0, 0.0, dist));
        t.look_at(pivot, Vec3::Y);
        if offset_x.abs() > 1e-4 {
            let half_w = dist * (22.5f32).to_radians().tan() * aspect;
            let right = t.rotation * Vec3::X;
            t.translation -= right * (half_w * 2.0 * offset_x);
        }
        t
    };
    //Is every point inside the frame with a margin? (station + satellite must both stay visible)
    let view = camera.logical_viewport_size().unwrap_or(Vec2::new(1920.0, 1080.0));
    let fits = |t: &Transform, pts: &[Vec3]| -> bool {
        let g = GlobalTransform::from(*t);
        pts.iter().all(|&p| match camera.world_to_viewport(&g, p) {
            Ok(v) => v.x > view.x * 0.08 && v.x < view.x * 0.92 && v.y > view.y * 0.10 && v.y < view.y * 0.90,
            Err(_) => false,
        })
    };
    //The globe's silhouette as seen from this view: four rim points (right/left/top/bottom of the disc)
    let r_earth = (cfg.scene.earth_radius_km / cfg.scene.km_per_unit) as f32;
    let globe_rim = |t: &Transform| -> [Vec3; 4] {
        let (r, u) = (t.rotation * Vec3::X * r_earth, t.rotation * Vec3::Y * r_earth);
        [r, -r, u, -u]
    };
    let globe_fits = |t: &Transform| -> bool {
        let g = GlobalTransform::from(*t);
        globe_rim(t).iter().all(|&p| match camera.world_to_viewport(&g, p) {
            Ok(v) => v.x > view.x * 0.02 && v.x < view.x * 0.98 && v.y > view.y * 0.02 && v.y < view.y * 0.98,
            Err(_) => false,
        })
    };

    //Where the camera wants to be. Selected: the camera pivots around the satellite itself, sitting outward of
    //it on the station's side so the cone and the line of sight read in perspective with the globe behind.
    //Cleared: back home (pivot on the globe's centre: the station-facing view, or wherever the camera was
    //before the pick).
    let st_pos = station.get_single().ok().map(|s| s.translation()).filter(|d| *d != Vec3::ZERO);
    let st_dir = st_pos.map(|p| p.normalize());
    let sat_pos = sel.0.and_then(|col| sats.iter().find(|(s, _)| s.0 == col)).map(|(_, t)| t.translation());
    let sat_dir = sat_pos.map(|p| p.normalize_or_zero());
    let selected = sat_dir.is_some();
    //The pivot: the selected satellite (it moves; the camera moves with it), else the globe's centre
    let want_pivot = if selected { sat_pos.unwrap_or(Vec3::ZERO) } else { Vec3::ZERO };
    //Distance is measured from the pivot: around a satellite the floor is far lower than around the globe
    let min_d = if selected { c.min_sat_distance } else { c.min_distance };
    if sel.is_changed() || selected != fly.selected {
        if selected && fly.home.is_none() { fly.home = Some((cam.yaw, cam.pitch, cam.distance)); }
        fly.from = (cam.yaw, cam.pitch, cam.distance);
        fly.from_pivot = cam.pivot;
        fly.t = 0.0;
        fly.selected = selected;
        fly.user_zoom = false;
        fly.hold = 0.0;   // a pick (or a clear) is a request to fly: it cancels the manual hold
    }
    let to_angles = |dir: Vec3| (dir.x.atan2(dir.z), dir.y.clamp(-0.999, 0.999).asin());
    let goal: Option<(f32, f32, f32)> = match (sat_dir, st_dir) {
        (Some(sd), Some(st)) => {
            //From the satellite, look-from direction = outward over the midpoint of station and satellite,
            //pushed sideways (their common normal) and lifted a little. Then back off: less side push and
            //more distance until the station, the satellite and the whole globe fit in the frame.
            let home_d = fly.home.map_or(cam.distance, |h| h.2);
            let base_d = (home_d * c.select_zoom).clamp(min_d, c.max_distance);
            let side = st.cross(sd).normalize_or_zero();
            let pts = [st_pos.unwrap_or(Vec3::ZERO), sat_pos.unwrap_or(Vec3::ZERO)];
            let mut pick = None;
            'search: for push in [0.55, 0.3, 0.0] {
                let dir = ((st + sd).normalize_or_zero() + side * push + Vec3::Y * 0.12).normalize_or_zero();
                let (y, p) = to_angles(dir);
                for k in 0..10 {
                    let d = (base_d * 1.12f32.powi(k)).min(c.max_distance);
                    let t = place(want_pivot, y, p, d);
                    if fits(&t, &pts) && globe_fits(&t) { pick = Some((y, p, d)); break 'search; }
                    if d >= c.max_distance { break; }
                }
            }
            Some(pick.unwrap_or_else(|| { let (y, p) = to_angles((st + sd).normalize_or_zero()); (y, p, c.max_distance) }))
        }
        (Some(sd), None) => { let (y, p) = to_angles(sd); Some((y, p, cam.distance)) }
        (None, _) => match fly.home {
            Some(h) => Some(if c.follow_station { st_dir.map_or(h, |st| { let (y, p) = to_angles(st); (y, p, h.2) }) } else { h }),
            None => if c.follow_station { st_dir.map(|st| { let (y, p) = to_angles(st); (y, p, cam.distance) }) } else { None },
        },
    };
    if dragging {
        for ev in motion.read() {
            cam.yaw -= ev.delta.x * c.drag_sensitivity;
            cam.pitch = (cam.pitch + ev.delta.y * c.drag_sensitivity).clamp(-1.5, 1.5);
        }
        fly.hold = c.manual_hold_seconds;   // the view is the user's for a while after the drag ends
        cam.pivot = want_pivot;             // still pivoting on the satellite as it moves
    } else if fly.hold > 0.0 {
        //Manual hold: the calculated view waits. When the hold runs out the glide back starts from here.
        motion.clear();
        cam.pivot = want_pivot;
        fly.hold -= time.delta_secs();
        if fly.hold <= 0.0 { fly.hold = 0.0; fly.from = (cam.yaw, cam.pitch, cam.distance); fly.from_pivot = cam.pivot; fly.t = 0.0; }
    } else {
        motion.clear();
        let lerp_angle = |a: f32, b: f32, e: f32| { let d = (b - a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI; a + d * e };
        if fly.t < 1.0 {
            fly.t = (fly.t + time.delta_secs() / fly_seconds.max(0.05)).min(1.0);
            let e = fly.t * fly.t * (3.0 - 2.0 * fly.t);   // smoothstep: eases out of the old view and into the new
            cam.pivot = fly.from_pivot.lerp(want_pivot, e);   // the pivot glides globe <-> satellite with the view
            if let Some((gy, gp, gd)) = goal {
                cam.yaw = lerp_angle(fly.from.0, gy, e);
                cam.pitch = fly.from.1 + (gp - fly.from.1) * e;
                cam.distance = fly.from.2 + (gd - fly.from.2) * e;
            }
        } else {
            //Arrived: keep tracking the goal as the satellite moves. Distance eases too, unless the user zoomed.
            cam.pivot = want_pivot;
            if let Some((gy, gp, gd)) = goal {
                cam.yaw = gy; cam.pitch = gp;
                if selected && !fly.user_zoom { cam.distance += (gd - cam.distance) * (1.0 - (-2.5 * time.delta_secs()).exp()); }
                if !selected { fly.home = None; }
            }
        }
    }
    for ev in wheel.read() {
        if primary.get(ev.window).is_err() || !wheel_here { continue; }   // scrolled in another window or another tile
        cam.distance = (cam.distance * (1.0 - ev.y * c.zoom_step)).clamp(min_d, c.max_distance);
        fly.user_zoom = true;
    }
    //Never let the globe leave the frame: back the camera off until its whole disc is on screen. Only while
    //the calculated view is in charge and has arrived: a drag, the hold after it, a glide in progress or a
    //hand zoom are the user's (around a satellite, turning the view puts the globe off-centre on purpose).
    let manual = fly.user_zoom || dragging || fly.hold > 0.0 || fly.t < 1.0;
    for _ in 0..60 {
        if manual || globe_fits(&place(cam.pivot, cam.yaw, cam.pitch, cam.distance)) || cam.distance >= c.max_distance { break; }
        cam.distance = (cam.distance * 1.03).min(c.max_distance);
    }
    let rot = Quat::from_rotation_y(cam.yaw) * Quat::from_rotation_x(-cam.pitch);
    tf.translation = cam.pivot + rot * Vec3::new(0.0, 0.0, cam.distance);
    //Orbiting a satellite can swing the camera round to the Earth's side of it: never go under the surface
    let floor = r_earth * 1.02;
    if tf.translation.length() < floor { tf.translation = tf.translation.normalize_or_zero() * floor; }
    tf.look_at(cam.pivot, Vec3::Y);
    //Slide the camera sideways (keeping its aim) so the globe lands at globe_offset_x across the window.
    if offset_x.abs() > 1e-4 {
        let half_w = cam.distance * (22.5f32).to_radians().tan() * aspect;
        let right = tf.rotation * Vec3::X;
        tf.translation -= right * (half_w * 2.0 * offset_x);
    }
}

//------------------------------------------------------------------------------------------ drawing
//Each satellite gets a short line from its current position to ahead_minutes ahead.
//The selected one is brighter and also gets a trail behind it, plus its ground track (the sub-satellite
//point over the same stretch, drawn on the globe in the Earth's turning frame) and a nadir line down to
//it; everything else dims while something is selected.
fn draw_orbits(orbits: Res<Orbits>, sel: Res<Selected>, sim: Res<Sim>, cat: Res<Catalog>, cfg: Res<Config>,
               ranks: Res<Ranks>, ranked_only: Res<RankedOnly>, regions: Res<Regions>, mut crossings: ResMut<Crossings>,
               cats: Res<Categories>, tfilter: Res<TypeFilter>,
               cam: Query<&GlobalTransform, With<Camera3d>>, earth: Query<&GlobalTransform, With<Earth>>,
               mut gizmos: Gizmos, mut bold: Gizmos<BoldLines>) {
    crossings.0.clear();
    let facing = cam.get_single().map(|c| c.rotation()).unwrap_or_default();
    let earth_rot = earth.get_single().map(|e| e.rotation()).unwrap_or_default();
    let r_ground = (cfg.scene.earth_radius_km / cfg.scene.km_per_unit) as f32 * 1.003;   // just above the surface
    let c_ground = hex(&cfg.colors.ground_track);
    let step = cfg.data.step_seconds;
    let ahead_cols = ((cfg.tracks.ahead_minutes as f64 * 60.0) / step).round() as usize;
    let sel_ahead_cols = ((cfg.tracks.selected_ahead_minutes.max(cfg.tracks.ahead_minutes) as f64 * 60.0) / step).round() as usize;
    let trail_cols = ((cfg.tracks.trail_minutes as f64 * 60.0) / step).round() as usize;
    let (c_orbit, c_dim, c_sel, c_trail, c_cone, c_aos, c_los) =
        (hex(&cfg.colors.orbit), hex(&cfg.colors.orbit_dim), hex(&cfg.colors.orbit_sel), hex(&cfg.colors.trail_sel),
         hex(&cfg.colors.track_in_cone), hex(&cfg.colors.aos), hex(&cfg.colors.los));
    let mark = cfg.scene.cross_size * 1.5;
    let st = &cfg.station;
    let sta = geodetic_to_ecef(st.lat_deg, st.lon_deg, st.alt_m);
    let region = regions.list.get(regions.current);
    let jd_now = sim.jd0 + sim.t / 86400.0;

    for (i, m) in orbits.0.iter().enumerate() {
        let t = sat_t(&sim, &cat, i);
        if t < 0.0 { continue; }
        let selected = sel.0 == Some(i);
        if (ranked_only.0 && !ranks.columns.contains(&i) || !cats.allows(&tfilter, i)) && !selected { continue; }
        let ranked = ranks.columns.contains(&i);
        //Focus: while something is picked every other track fades to the dim colour, ranked or not
        let color = match (sel.0, top_tier(&ranks, i)) {
            (Some(_), _) if selected => c_sel,
            (Some(_), _) => c_dim,
            (None, Some((_, k))) => scaled(hex(&cfg.colors.rank_top), 0.9 * k),
            (None, None) => c_orbit,
        };
        //Only the ranked set and the pick get a track, and theirs is a whole orbit (ranked_orbits periods);
        //everyone else is just a marker
        if !(selected || ranked) && ahead_cols == 0 { continue; }
        let last = m.ncols() - 1;
        let cur = ((t / step) as usize).min(last);
        let span_cols = if selected || ranked {
            match (cfg.tracks.ranked_orbits > 0.0, sat_period_s(&cfg, m, t)) {
                (true, Some(period)) => ((period * cfg.tracks.ranked_orbits) / step).round() as usize,
                _ => sel_ahead_cols,
            }
        } else { ahead_cols };
        let end = (cur + span_cols).min(last);

        //Points ahead: the live position, then every stored column up to `end`
        let mut ahead: Vec<Vec3> = Vec::with_capacity(end - cur + 2);
        ahead.push(sat_position(&cfg, m, t));
        ahead.extend((cur + 1..=end).map(|c| column(&cfg, m, c)));

        //The picked and the ranked satellites: light up the stretch of track that lies inside the cone.
        //Each column is checked from the station at its own time (the Earth turns under the track).
        //Cone crossings: for the pick, or for the ranked set while nothing is picked
        if (selected || (ranked && sel.0.is_none())) && region.is_some() {
            let region = region.unwrap();
            let inside = |k: usize| -> bool {
                let (eci, dt) = if k == 0 { (sat_eci_km(&cfg, m, t), 0.0) }
                                else { let c = cur + k; ([m[(0, c)], m[(1, c)], m[(2, c)]], c as f64 * step - t) };
                let ecef = eci_to_ecef(eci, gmst_rad(jd_now + dt / 86400.0));
                let (az, el, _) = look_angles(sta, st.lat_deg, st.lon_deg, ecef);
                region.contains(az, el)
            };
            let flags: Vec<bool> = (0..ahead.len()).map(inside).collect();
            //Split into runs; a run inside the cone is drawn blue and bold
            let mut k = 0;
            while k + 1 < ahead.len() {
                let on = flags[k] && flags[k + 1];
                let mut j = k + 1;
                while j + 1 < ahead.len() && (flags[j] && flags[j + 1]) == on { j += 1; }
                if on { bold.linestrip(ahead[k..=j].iter().copied(), c_cone); }
                else if selected { bold.linestrip(ahead[k..=j].iter().copied(), color); }   // the pick's track is bold throughout
                else { gizmos.linestrip(ahead[k..=j].iter().copied(), color); }
                k = j;
            }
            //Edge crossings, midway between the two samples. Vector-display style, camera facing:
            //  AOS (comes into view): a diamond with a tick pointing along the track
            //  LOS (drops out of view): a diamond with a bar across it
            let (right, up) = (facing * Vec3::X, facing * Vec3::Y);
            for k in 0..ahead.len().saturating_sub(1) {
                if flags[k] != flags[k + 1] {
                    let entering = flags[k + 1];
                    let p = (ahead[k] + ahead[k + 1]) * 0.5;
                    let col = if entering { c_aos } else { c_los };
                    let d = [p + up * mark, p + right * mark, p - up * mark, p - right * mark, p + up * mark];
                    bold.linestrip(d, col);
                    if entering {
                        let along = (ahead[k + 1] - ahead[k]).normalize_or_zero();
                        bold.line(p + along * mark * 1.2, p + along * mark * 2.6, col);
                    } else {
                        bold.line(p - right * mark * 1.6, p + right * mark * 1.6, col);
                    }
                    crossings.0.push((p, entering));
                }
            }
        } else if selected {
            bold.linestrip(ahead, color);
        } else {
            gizmos.linestrip(ahead, color);
        }

        if selected {
            let start = cur.saturating_sub(trail_cols);
            let mut trail: Vec<Vec3> = (start..=cur).map(|c| column(&cfg, m, c)).collect();
            trail.push(sat_position(&cfg, m, t));
            bold.linestrip(trail, c_trail);

            //Ground track: each sample's sub-satellite point at its own time. Inertial -> Earth-fixed by the
            //sidereal angle of that moment, dropped onto the sphere, then turned with the globe as it spins
            //(the same frame the station dot and the coastlines live in). Trail and look-ahead in one strip.
            if c_ground.alpha() > 0.0 {
                let sub = |eci: [f64; 3], dt: f64| -> Vec3 {
                    let ecef = eci_to_ecef(eci, gmst_rad(jd_now + dt / 86400.0));
                    earth_rot * (to_scene(&cfg, ecef[0], ecef[1], ecef[2]).normalize_or_zero() * r_ground)
                };
                let at_col = |c: usize| sub([m[(0, c)], m[(1, c)], m[(2, c)]], c as f64 * step - t);
                let here = sub(sat_eci_km(&cfg, m, t), 0.0);
                let mut ground: Vec<Vec3> = Vec::with_capacity(end - start + 2);
                ground.extend((start..=cur).map(at_col));
                ground.push(here);
                ground.extend((cur + 1..=end).map(at_col));
                gizmos.linestrip(ground, c_ground);
                gizmos.line(sat_position(&cfg, m, t), here, c_ground);   // nadir: satellite straight down to its footprint
            }
        }
    }
}

//True when the globe sits between the camera and the point
fn behind_globe(cam_pos: Vec3, p: Vec3, r_earth: f32) -> bool {
    let d = p - cam_pos;
    let len = d.length();
    if len < 1e-6 { return false; }
    let dir = d / len;
    let t = -cam_pos.dot(dir);
    if t < 0.0 || t > len { return false; }
    let closest = cam_pos + dir * t;
    closest.length() < r_earth
}

//Place the "1 NAME" .. "3 NAME" tags beside the top-ranked markers (hidden when the globe is in the way)
fn update_rank_tags(
    ranks: Res<Ranks>,
    cfg: Res<Config>,
    sel: Res<Selected>,
    cat: Res<Catalog>,
    cam_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    sats: Query<(&Satellite, &GlobalTransform, &Visibility)>,
    mut tags: Query<(&RankTag, &mut Node, &mut Visibility, &Children), Without<Satellite>>,
    mut texts: Query<&mut Text>,
    ui_scale: Res<UiScale>,
) {
    let Ok((camera, cam_tf)) = cam_q.get_single() else { return };
    for (tag, mut node, mut vis, children) in &mut tags {
        //Tag 0 follows the pick (unless it is already carrying a top-3 tag); tags 1-3 follow the ranks
        let (column, label) = if tag.0 == 0 {
            match sel.0 {
                Some(col) if !ranks.entries.iter().any(|e| e.pass.column == col && e.rank <= 3) => {
                    let id = cat.ids.get(col).copied().flatten();
                    (col, id.and_then(|i| cat.names.get(&i).cloned()).unwrap_or_else(|| id.map_or(format!("TRACK #{col}"), |i| format!("NORAD {i}"))))
                }
                _ => { *vis = Visibility::Hidden; continue }
            }
        } else {
            let Some(e) = ranks.entries.iter().find(|e| e.rank == tag.0) else { *vis = Visibility::Hidden; continue };
            (e.pass.column, format!("{} {}", tag.0, e.name))
        };
        let Some((_, tf, svis)) = sats.iter().find(|(s, _, _)| s.0 == column) else { *vis = Visibility::Hidden; continue };
        if *svis == Visibility::Hidden { *vis = Visibility::Hidden; continue; }
        let r_earth = (cfg.scene.earth_radius_km / cfg.scene.km_per_unit) as f32;
        if behind_globe(cam_tf.translation(), tf.translation(), r_earth) { *vis = Visibility::Hidden; continue; }
        match camera.world_to_viewport(cam_tf, tf.translation()) {
            Ok(p) => {
                let p = p / ui_scale.0.max(0.01);
                node.left = Val::Px(p.x + 18.0);
                node.top = Val::Px(p.y - 26.0);
                *vis = Visibility::Inherited;
                for child in children.iter() {
                    if let Ok(mut t) = texts.get_mut(*child) {
                        let want = label.to_uppercase();
                        if t.0 != want { t.0 = want; }
                    }
                }
            }
            Err(_) => *vis = Visibility::Hidden,
        }
    }
}

//UPDATE TLES notice follows Sim.exhausted; the block cursor blinks once a second
fn update_exhausted(sim: Res<Sim>, time: Res<Time>,
                    mut boxq: Query<&mut Visibility, With<ExhaustedBox>>,
                    mut cursor: Query<&mut TextColor, With<ExhaustedCursor>>) {
    let want = if sim.exhausted { Visibility::Inherited } else { Visibility::Hidden };
    for mut v in &mut boxq { if *v != want { *v = want; } }
    if !sim.exhausted { return; }
    let on = time.elapsed_secs_f64() % 1.0 < 0.5;
    for mut c in &mut cursor { let mut s = c.0.to_srgba(); s.alpha = if on { 1.0 } else { 0.0 }; c.0 = Color::Srgba(s); }
}

//Explore: pick the next satellite when the dwell time is up (any live track, avoiding recent picks)
fn explore_tick(time: Res<Time>, cfg: Res<Config>, orbits: Res<Orbits>, sim: Res<Sim>, cat: Res<Catalog>,
                mut explore: ResMut<Explore>, mut sel: ResMut<Selected>, cats: Res<Categories>, tfilter: Res<TypeFilter>) {
    if !explore.on { return; }
    let now = time.elapsed_secs_f64();
    if now < explore.next { return; }
    explore.next = now + cfg.camera.explore_seconds.max(3.0) as f64;
    let n = orbits.0.len();
    if n == 0 { return; }
    for _ in 0..64 {
        let i = (explore.rand() % n as u64) as usize;
        if sat_t(&sim, &cat, i) < 0.0 || explore.recent.contains(&i) || !cats.allows(&tfilter, i) { continue; }
        explore.recent.push(i);
        if explore.recent.len() > 32 { explore.recent.remove(0); }
        sel.0 = Some(i);
        return;
    }
}

//Place the AOS / LOS labels beside this frame's view-edge crossings
fn update_cross_tags(
    crossings: Res<Crossings>,
    cfg: Res<Config>,
    cam_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut tags: Query<(&CrossTag, &mut Node, &mut Visibility, &Children)>,
    mut texts: Query<(&mut Text, &mut TextColor)>,
    ui_scale: Res<UiScale>,
) {
    let Ok((camera, cam_tf)) = cam_q.get_single() else { return };
    let r_earth = (cfg.scene.earth_radius_km / cfg.scene.km_per_unit) as f32;
    let (c_aos, c_los) = (hex(&cfg.colors.aos), hex(&cfg.colors.los));
    for (tag, mut node, mut vis, children) in &mut tags {
        let Some(&(p, entering)) = crossings.0.get(tag.0) else { *vis = Visibility::Hidden; continue };
        if behind_globe(cam_tf.translation(), p, r_earth) { *vis = Visibility::Hidden; continue; }
        match camera.world_to_viewport(cam_tf, p) {
            Ok(v) => {
                let v = v / ui_scale.0.max(0.01);
                node.left = Val::Px(v.x + 12.0);
                node.top = Val::Px(v.y - 20.0);
                *vis = Visibility::Inherited;
                for child in children.iter() {
                    if let Ok((mut t, mut c)) = texts.get_mut(*child) {
                        let want = if entering { "AOS" } else { "LOS" };
                        if t.0 != want { t.0 = want.into(); }
                        let col = if entering { c_aos } else { c_los };
                        if c.0 != col { c.0 = col; }
                    }
                }
            }
            Err(_) => *vis = Visibility::Hidden,
        }
    }
}

//Vector globe: graticule circles and continent polylines as glowing lines, turning with the Earth frame
fn draw_vector_globe(
    cfg: Res<Config>,
    outlines: Res<Outlines>,
    earth: Query<&GlobalTransform, With<Earth>>,
    mut gizmos: Gizmos,
    mut bold: Gizmos<BoldLines>,
) {
    if !cfg.scene.vector_globe { return; }
    let Ok(etf) = earth.get_single() else { return };
    let rot = etf.rotation();
    let r = (cfg.scene.earth_radius_km / cfg.scene.km_per_unit) as f32 * 1.002;
    let major = hex(&cfg.colors.grid_major);
    let minor = hex(&cfg.colors.grid_minor);
    let outline = hex(&cfg.colors.outline);

    //Parallels every 15 deg (equator brighter): circles about the pole axis
    let flat = Quat::from_rotation_arc(Vec3::Z, Vec3::Y);
    for k in -5..=5 {
        let lat = (k as f32 * 15.0).to_radians();
        let center = rot * Vec3::new(0.0, r * lat.sin(), 0.0);
        gizmos.circle(Isometry3d::new(center, rot * flat), r * lat.cos(), if k == 0 { major } else { minor });
    }
    //Meridians every 15 deg: great circles whose normal lies in the equatorial plane
    for k in 0..12 {
        let lon = (k as f32 * 15.0).to_radians();
        let n = Vec3::new(-lon.sin(), 0.0, -lon.cos());       // ECEF (-sin, cos, 0) through the scene axis swap
        let q = rot * Quat::from_rotation_arc(Vec3::Z, n);
        gizmos.circle(Isometry3d::new(Vec3::ZERO, q), r, if k == 0 { major } else { minor });
    }
    //Continents: thicker, full-brightness lines so the coastlines read from across the room
    for ring in &outlines.0 {
        bold.linestrip(ring.iter().map(|p| rot * *p), outline);
    }
}

//Cross markers (vector display look): a small camera-facing + per satellite, bigger and brighter when in view
fn draw_cross_markers(
    cfg: Res<Config>,
    sel: Res<Selected>,
    ranks: Res<Ranks>,
    sats: Query<(&Satellite, &GlobalTransform, &Visibility, &SatInView)>,
    cam: Query<&GlobalTransform, With<Camera3d>>,
    mut gizmos: Gizmos,
) {
    if !cfg.scene.marker_style.eq_ignore_ascii_case("cross") { return; }
    let Ok(cam_tf) = cam.get_single() else { return };
    let right = cam_tf.rotation() * Vec3::X;
    let up = cam_tf.rotation() * Vec3::Y;
    let (c_norm, c_view, c_sel, c_top) = (hex(&cfg.colors.marker), hex(&cfg.colors.marker_in_view), hex(&cfg.colors.selected), hex(&cfg.colors.rank_top));
    for (sat, tf, vis, in_view) in &sats {
        if *vis == Visibility::Hidden { continue; }
        let selected = sel.0 == Some(sat.0);
        let tier = top_tier(&ranks, sat.0);
        let (color, size) = if selected { (c_sel, cfg.scene.cross_size * 1.8) }
                            else if sel.0.is_some() { (scaled(c_norm, FOCUS_DIM), cfg.scene.cross_size * 0.8) }   // focus: everyone else fades
                            else if let Some((rank, k)) = tier { (scaled(c_top, k), cfg.scene.cross_size * (2.6 - 0.4 * rank as f32)) }
                            else if in_view.0 { (c_view, cfg.scene.cross_size * 1.4) }
                            else { (c_norm, cfg.scene.cross_size) };
        let p = tf.translation();
        gizmos.line(p - right * size, p + right * size, color);
        gizmos.line(p - up * size, p + up * size, color);
    }
}

//Small gold ring around every ranked satellite so they stand out in the crowd
fn draw_rank_rings(
    ranks: Res<Ranks>,
    cfg: Res<Config>,
    sel: Res<Selected>,
    sats: Query<(&Satellite, &GlobalTransform, &Visibility)>,
    cam: Query<&GlobalTransform, With<Camera3d>>,
    mut gizmos: Gizmos,
) {
    if ranks.columns.is_empty() { return; }
    let Ok(cam_tf) = cam.get_single() else { return };
    //Focus: with a pick, only its own ring stays bright
    let dim_others = sel.0.is_some();
    let c = hex(&cfg.colors.rank);
    let top = hex(&cfg.colors.rank_top);
    let facing = cam_tf.rotation();
    for (sat, tf, vis) in &sats {
        if *vis == Visibility::Hidden || !ranks.columns.contains(&sat.0) { continue; }
        let p = tf.translation();
        let fade = if dim_others && sel.0 != Some(sat.0) { FOCUS_DIM } else { 1.0 };
        let (c, top) = (scaled(c, fade), scaled(top, fade));
        match top_tier(&ranks, sat.0) {
            Some((rank, k)) => {
                let r = cfg.scene.reticle_radius * (1.25 - 0.2 * rank as f32);   // 1.05, 0.85, 0.65
                let col = scaled(top, k);
                gizmos.circle(Isometry3d::new(p, facing), r, col);
                if rank <= 2 { gizmos.circle(Isometry3d::new(p, facing), r * 0.72, col); }
                if rank == 1 {
                    for q in 0..4 {
                        let ang = q as f32 * std::f32::consts::FRAC_PI_2;
                        let dir = facing * Vec3::new(ang.cos(), ang.sin(), 0.0);
                        gizmos.line(p + dir * r * 1.1, p + dir * r * 1.6, col);
                    }
                }
            }
            None => { gizmos.circle(Isometry3d::new(p, facing), cfg.scene.reticle_radius * 0.6, c); }
        }
    }
}

//Camera-facing ring around the station, and a reticle with tick marks around the selected satellite.
fn draw_reticles(
    sel: Res<Selected>,
    cfg: Res<Config>,
    sats: Query<(&Satellite, &GlobalTransform)>,
    station: Query<&GlobalTransform, With<StationDot>>,
    cam: Query<&GlobalTransform, With<Camera3d>>,
    mut gizmos: Gizmos,
    mut bold: Gizmos<BoldLines>,
) {
    let Ok(cam_tf) = cam.get_single() else { return };
    //Station: a bold camera-facing ring with four ticks, so home is unmistakable
    if let Ok(st) = station.get_single() {
        let col = hex(&cfg.colors.station);
        let (p, r) = (st.translation(), cfg.scene.station_size * 2.5);
        bold.circle(Isometry3d::new(p, cam_tf.rotation()), r, col);
        let (right, up) = (cam_tf.rotation() * Vec3::X, cam_tf.rotation() * Vec3::Y);
        for d in [right, -right, up, -up] { bold.line(p + d * r * 1.2, p + d * r * 2.2, col); }
        //Line of sight from the station to the selected satellite
        if let Some((_, sat_tf)) = sel.0.and_then(|col| sats.iter().find(|(s, _)| s.0 == col)) {
            let mut c = hex(&cfg.colors.reticle_sel).to_srgba(); c.alpha = 0.85;
            bold.line(p, sat_tf.translation(), Color::Srgba(c));
        }
    }
    let Some(i) = sel.0 else { return };
    let Some((_, tf)) = sats.iter().find(|(s, _)| s.0 == i) else { return };
    let facing = cam_tf.rotation();
    let p = tf.translation();
    let r = cfg.scene.reticle_radius;
    let c = hex(&cfg.colors.reticle_sel);
    gizmos.circle(Isometry3d::new(p, facing), r, c);
    for k in 0..4 {
        let ang = k as f32 * std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_4;
        let dir = facing * Vec3::new(ang.cos(), ang.sin(), 0.0);
        gizmos.line(p + dir * r * 1.18, p + dir * r * 1.62, c);
    }
}
