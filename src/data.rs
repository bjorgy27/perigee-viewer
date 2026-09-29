/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Data source: where the viewer gets Perigee's files from - the folder next to ORBIT_DATA.json.
/////////////////////////////////////////////////////////////////////////////////////////////////////////
use nalgebra::Matrix6xX;
use perigee_orbit::ElSetMatrix;

pub struct Source { pub dir: std::path::PathBuf, pub orbit_file: std::path::PathBuf }

pub struct Loaded {
    pub orbits: Vec<Matrix6xX<f64>>,
    pub sorted_sats: Option<String>,
    pub elset: Option<String>,
    pub transmitters: Option<String>,
    pub ranks: Option<String>,
    pub categories: Option<String>,   // CATEGORIES.json: satellite types for the TYPE filter (optional)
}

impl Source {
    pub fn describe(&self) -> String { self.orbit_file.display().to_string() }

    fn read(&self, name: &str) -> Option<String> { std::fs::read_to_string(self.dir.join(name)).ok() }

    pub fn load(&self) -> Result<Loaded, String> {
        let orbit_file = &self.orbit_file;
        let text = std::fs::read_to_string(orbit_file).map_err(|e| format!("{}: {e}", orbit_file.display()))?;
        let orbits: Vec<Matrix6xX<f64>> = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", orbit_file.display()))?;
        Ok(Loaded {
            orbits,
            sorted_sats: self.read("SORTED_SATS.json"),
            elset: self.read("ELSET.json"),
            transmitters: self.read("NORADs.json"),
            ranks: self.read("SATELLITE_RANKS.json"),
            categories: self.read("CATEGORIES.json"),
        })
    }

    /// Local propagation: the raw element sets (SORTED_SATS.json, true epochs) plus the same side files.
    /// `orbits` comes back empty; the viewer integrates them itself.
    pub fn load_elsets(&self) -> Result<(ElSetMatrix, Loaded), String> {
        let p = self.dir.join("SORTED_SATS.json");
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let rest = Loaded {
            orbits: Vec::new(),
            sorted_sats: None,
            elset: self.read("ELSET.json"),
            transmitters: self.read("NORADs.json"),
            ranks: self.read("SATELLITE_RANKS.json"),
            categories: self.read("CATEGORIES.json"),
        };
        let coe: ElSetMatrix = serde_json::from_str(&text).map_err(|e| format!("element sets: {e}"))?;
        if coe.ncols() == 0 { return Err("element sets: no satellites".into()); }
        Ok((coe, Loaded { sorted_sats: Some(text), ..rest }))
    }

    /// Tell Perigee which sky window to rank for: write VIEW_REGION.json next to the data.
    pub fn send_region(&self, json: &str) -> Result<(), String> {
        std::fs::write(self.dir.join("VIEW_REGION.json"), json).map_err(|e| e.to_string())
    }

    /// Latest rankings text
    pub fn fetch_ranks(&self) -> Option<String> { self.read("SATELLITE_RANKS.json") }
}
