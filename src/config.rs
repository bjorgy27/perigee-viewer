/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Viewer settings loaded from viewer.toml (see that file for what each key does).
/// Every section and key has a default, so a partial or missing file still works.
/////////////////////////////////////////////////////////////////////////////////////////////////////////
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Config {
    pub data: Data,
    pub station: Station,
    pub sim: Sim,
    pub tracks: Tracks,
    pub scene: Scene,
    pub earth_grid: EarthGrid,
    pub camera: CameraCfg,
    pub lighting: Lighting,
    pub colors: Colors,
    pub fx: Fx,
    pub perf: Perf,
    pub regions: Vec<Region>,     // sky windows the dish can use; the first is the default
}

//A window of sky: azimuth from az_from clockwise to az_to (degrees from north), elevation el_min..el_max.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Region { pub name: String, pub az_from: f64, pub az_to: f64, pub el_min: f64, pub el_max: f64 }
impl Default for Region {
    fn default() -> Self { Self { name: "FULL SKY".into(), az_from: 0.0, az_to: 360.0, el_min: 10.0, el_max: 90.0 } }
}
impl Region {
    pub fn contains(&self, az_deg: f64, el_deg: f64) -> bool {
        if el_deg < self.el_min || el_deg > self.el_max { return false; }
        let span = (self.az_to - self.az_from).rem_euclid(360.0);
        if span == 0.0 || (self.az_to - self.az_from).abs() >= 360.0 { return true; }
        (az_deg - self.az_from).rem_euclid(360.0) <= span
    }
    pub fn full_azimuth(&self) -> bool { (self.az_to - self.az_from).abs() >= 360.0 || (self.az_to - self.az_from).rem_euclid(360.0) == 0.0 }
}

//Rendering cost knobs
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Perf { pub msaa: bool, pub hdr: bool, pub tonemapping: String, pub present_mode: String, pub pipelined_rendering: bool }
impl Default for Perf {
    fn default() -> Self { Self { msaa: true, hdr: true, tonemapping: "tony".into(), present_mode: "vsync".into(), pipelined_rendering: true } }
}

//Screen effects
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Fx { pub scanlines: f32, pub scanline_period_px: u32, pub vignette: f32, pub cursor_blink: bool }
impl Default for Fx {
    fn default() -> Self { Self { scanlines: 0.0, scanline_period_px: 3, vignette: 0.0, cursor_blink: false } }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Data {
    pub orbit_file: String, pub step_seconds: f64, pub ranks_poll_seconds: f64,
    pub top_ranked: usize,        // how many of Perigee's ranked passes the viewer keeps (panel rows, rings, ranked-only filter)
    pub rerank_command: String,   // Perigee binary to run for a fresh ranking ("" disables)
    pub rerank_seconds: f64,      // how often to run it
    //Local propagation: integrate the element sets (SORTED_SATS.json) here with the engine's own RK4,
    //instead of loading the vectors the engine wrote. No 200 MB file, and the data never runs out.
    pub local_propagation: bool,      // false: read orbit_file like before
    pub propagation_budget_ms: f64,   // time slice per frame spent integrating (the rest of the frame is drawing)
    pub propagation_threads: usize,   // 0 = one per core
    pub keep_before_min: f64,         // minutes of track kept before launch time (trails, History mode)
    pub propagate_ahead_h: f64,       // hours past now to integrate to
    pub propagate_min_ahead_h: f64,   // when the shortest track has less than this left, extend everyone to propagate_ahead_h
}
impl Default for Data {
    fn default() -> Self { Self {
        orbit_file: "../Perigee/src/ORBIT_DATA.json".into(), step_seconds: 60.0, ranks_poll_seconds: 60.0, top_ranked: 7,
        rerank_command: "../Perigee/target/release/perigee".into(), rerank_seconds: 120.0,
        local_propagation: true, propagation_budget_ms: 50.0, propagation_threads: 0,
        keep_before_min: 45.0, propagate_ahead_h: 12.0, propagate_min_ahead_h: 6.0,
    } }
}

//Your ground station. Latitude north-positive, longitude east-positive (west is negative).
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Station {
    pub auto_locate: bool,         // true: look the position up by IP (ipinfo.io) at launch, like the radar widget
    pub name: String, pub lat_deg: f64, pub lon_deg: f64, pub alt_m: f64,
    pub elevation_mask_deg: f64,   // satellites below this elevation count as out of view (FULL SKY region floor)
    pub view_range_km: f64,        // how far out the drawn view cone reaches (slant range)
    pub cone_range_km: f64,        // the translucent cone body runs out to this range, fading to nothing
    pub cone_fade_power: f32,      // how quickly it fades: alpha = (1 - dist/cone_range)^power
    pub region: String,            // name of the region selected at launch ("" = the first in [[regions]])
}
impl Default for Station {
    fn default() -> Self { Self {
        auto_locate: true,
        name: "STATION".into(), lat_deg: 40.7128, lon_deg: -74.0060, alt_m: 10.0,
        elevation_mask_deg: 10.0, view_range_km: 3000.0, cone_range_km: 9000.0, cone_fade_power: 3.0, region: String::new(),
    } }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Sim { pub start_mode: String, pub start_speed: f64, pub max_speed: f64, pub min_speed: f64 }
impl Default for Sim {
    fn default() -> Self { Self { start_mode: "live".into(), start_speed: 60.0, max_speed: 86400.0, min_speed: 1.0 } }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Tracks { pub ahead_minutes: usize, pub trail_minutes: usize, pub pick_radius_px: f32,
    pub selected_ahead_minutes: usize,   // fallback length for the pick / ranked when the period cannot be worked out
    pub ranked_orbits: f64 }             // ranked satellites and the pick draw this many full orbits ahead (0 = use minutes)
impl Default for Tracks {
    fn default() -> Self { Self { ahead_minutes: 0, trail_minutes: 45, pick_radius_px: 18.0, selected_ahead_minutes: 60, ranked_orbits: 1.0 } }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Scene {
    pub km_per_unit: f64, pub earth_radius_km: f64, pub atmosphere_scale: f32,
    pub marker_size: f32, pub selected_scale: f32, pub marker_spin: f32, pub reticle_radius: f32,
    pub star_count: usize, pub star_distance: f32, pub star_size: f32,
    pub station_size: f32,
    pub earth_unlit: bool,      // true: no shading, the globe is exactly the texture (vector-display look)
    pub vector_globe: bool,     // true: graticule + continents drawn as glowing 3D lines instead of a texture
    pub marker_style: String,   // "cross" (vector display) or "diamond" (mesh)
    pub cross_size: f32,        // half-length of a cross marker, scene units
    pub outline_line_width: f32, // pixel width of the continent outlines and the station marker (vector globe)
}
impl Default for Scene {
    fn default() -> Self { Self {
        outline_line_width: 2.5,
        km_per_unit: 1000.0, earth_radius_km: 6378.0, atmosphere_scale: 1.035,
        marker_size: 0.07, selected_scale: 1.8, marker_spin: 1.2, reticle_radius: 0.34,
        star_count: 500, star_distance: 180.0, star_size: 0.12,
        station_size: 0.05,
        earth_unlit: false,
        vector_globe: false,
        marker_style: "diamond".into(),
        cross_size: 0.09,
    } }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct EarthGrid {
    pub major_deg: f32, pub minor_deg: f32, pub major_width: f32, pub minor_width: f32, pub texture_width: u32,
    pub outline_file: String, pub outline_width_px: u32,
}
impl Default for EarthGrid {
    fn default() -> Self { Self {
        major_deg: 90.0, minor_deg: 15.0, major_width: 0.18, minor_width: 0.07, texture_width: 2048,
        outline_file: "assets/countries.geojson".into(), outline_width_px: 1,
    } }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct CameraCfg { pub start_distance: f32, pub start_pitch: f32, pub min_distance: f32, pub max_distance: f32, pub drag_sensitivity: f32, pub zoom_step: f32, pub globe_offset_x: f32, pub follow_station: bool,
    pub select_zoom: f32, pub fly_seconds: f32,   // on a pick: zoom factor on the current distance, and the glide time
    pub manual_hold_seconds: f32,                 // after a drag: how long the view stays put before the calculated view glides back
    pub min_sat_distance: f32,                    // closest the camera comes to a selected satellite (world units; it pivots around it)
    pub explore_seconds: f32 }                    // explore mode: dwell time on each satellite before gliding to the next
impl Default for CameraCfg {
    fn default() -> Self { Self { start_distance: 28.0, start_pitch: 0.3, min_distance: 6.7, max_distance: 120.0, drag_sensitivity: 0.005, zoom_step: 0.1, globe_offset_x: 0.0, follow_station: false, select_zoom: 0.7, fly_seconds: 1.6, explore_seconds: 15.0, manual_hold_seconds: 10.0, min_sat_distance: 0.15 } }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Lighting { pub sun_illuminance: f32, pub sun_direction: [f32; 3], pub ambient_brightness: f32, pub bloom_intensity: f32 }
impl Default for Lighting {
    fn default() -> Self { Self { sun_illuminance: 12000.0, sun_direction: [80.0, 25.0, 40.0], ambient_brightness: 60.0, bloom_intensity: 0.25 } }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Colors {
    pub space: String, pub earth: String, pub grid_major: String, pub grid_minor: String, pub atmosphere: String,
    pub orbit: String, pub orbit_dim: String, pub orbit_sel: String, pub trail_sel: String, pub reticle_sel: String,
    pub marker: String, pub marker_glow: [f32; 3], pub marker_in_view_glow: [f32; 3],
    pub selected: String, pub selected_glow: [f32; 3],
    pub outline: String, pub station: String, pub view_cone: String, pub marker_in_view: String, pub rank: String, pub rank_top: String,
    pub track_in_cone: String, pub aos: String, pub los: String,   // track stretch inside the view; AOS / LOS marks
    pub ground_track: String,                                      // the pick's sub-satellite path on the globe + nadir line (alpha 00 hides)
    pub text: String, pub text_dim: String, pub button: String, pub button_hover: String, pub button_border: String,
}
impl Default for Colors {
    fn default() -> Self { Self {
        space: "#03030A".into(), earth: "#080E18".into(), grid_major: "#288CB4".into(), grid_minor: "#1A4E68".into(),
        atmosphere: "#40B3FF1A".into(),
        orbit: "#59D9FF29".into(), orbit_dim: "#59D9FF09".into(), orbit_sel: "#FF3B3BCC".into(),
        trail_sel: "#FF6060".into(), reticle_sel: "#FF4040".into(),
        marker: "#99F2FF".into(), marker_glow: [0.6, 2.4, 3.0], marker_in_view_glow: [0.8, 3.0, 1.2],
        selected: "#FF2A2A".into(), selected_glow: [6.0, 0.4, 0.4],
        outline: "#4FB8D9".into(), station: "#FF8C3A".into(),
        track_in_cone: "#4FA8FF".into(), aos: "#7CFFA6".into(), los: "#FF6A3A".into(),
        ground_track: "#FF606080".into(),
        view_cone: "#FF8C3A1F".into(), marker_in_view: "#7CFF9E".into(), rank: "#FFD24D".into(), rank_top: "#FFFFFF".into(),
        text: "#8CD9FF".into(), text_dim: "#4D7399".into(),
        button: "#1A40668C".into(), button_hover: "#2666994D".into(), button_border: "#59D9FF99".into(),
    } }
}

impl Config {
    //Load viewer.toml from the working directory. Missing file => defaults. A malformed file is an error
    //worth stopping on, since silently ignoring it would be confusing.
    pub fn load(path: &str) -> Config {
        match std::fs::read_to_string(path) {
            Ok(txt) => toml::from_str(&txt).unwrap_or_else(|e| panic!("{path}: {e}")),
            Err(_) => {
                println!("{path} not found, using built-in defaults");
                Config::default()
            }
        }
    }
}

//"#RRGGBB" or "#RRGGBBAA" -> Color. Anything unparsable becomes magenta so it's obvious.
pub fn hex(s: &str) -> Color {
    let h = s.trim().trim_start_matches('#');
    let byte = |i: usize| u8::from_str_radix(h.get(i..i + 2).unwrap_or("zz"), 16).ok();
    match (h.len(), byte(0), byte(2), byte(4)) {
        (6, Some(r), Some(g), Some(b)) => Color::srgb_u8(r, g, b),
        (8, Some(r), Some(g), Some(b)) => Color::srgba_u8(r, g, b, byte(6).unwrap_or(255)),
        _ => { eprintln!("bad color {s:?}"); Color::srgb(1.0, 0.0, 1.0) }
    }
}

//Same, but as [u8; 4] for writing texture pixels
pub fn hex_rgba8(s: &str) -> [u8; 4] {
    let c = hex(s).to_srgba();
    [(c.red * 255.0) as u8, (c.green * 255.0) as u8, (c.blue * 255.0) as u8, (c.alpha * 255.0) as u8]
}
