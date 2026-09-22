/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Data source: where the viewer gets Perigee's files from.
///   Files  - the folder next to ORBIT_DATA.json on this machine (desktop)
///   Remote - the Perigee cast server over HTTPS with a pinned self-signed certificate and a token (TV)
/// Both hand back the same texts / matrices, so everything downstream is identical.
/////////////////////////////////////////////////////////////////////////////////////////////////////////
use nalgebra::Matrix6xX;
use perigee_orbit::ElSetMatrix;
use std::sync::Arc;

pub enum Source {
    Files { dir: std::path::PathBuf, orbit_file: std::path::PathBuf },
    Remote { base: String, token: String, cert_sha256: Vec<u8> },
}

pub struct Loaded {
    pub orbits: Vec<Matrix6xX<f64>>,
    pub sorted_sats: Option<String>,
    pub elset: Option<String>,
    pub transmitters: Option<String>,
    pub ranks: Option<String>,
    pub categories: Option<String>,   // CATEGORIES.json: satellite types for the TYPE filter (optional)
}

impl Source {
    pub fn describe(&self) -> String {
        match self {
            Source::Files { orbit_file, .. } => orbit_file.display().to_string(),
            Source::Remote { base, .. } => format!("{base}/<token>/data/"),
        }
    }

    pub fn load(&self) -> Result<Loaded, String> {
        match self {
            Source::Files { dir, orbit_file } => {
                let text = std::fs::read_to_string(orbit_file).map_err(|e| format!("{}: {e}", orbit_file.display()))?;
                let orbits: Vec<Matrix6xX<f64>> = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", orbit_file.display()))?;
                Ok(Loaded {
                    orbits,
                    sorted_sats: std::fs::read_to_string(dir.join("SORTED_SATS.json")).ok(),
                    elset: std::fs::read_to_string(dir.join("ELSET.json")).ok(),
                    transmitters: std::fs::read_to_string(dir.join("NORADs.json")).ok(),
                    ranks: std::fs::read_to_string(dir.join("SATELLITE_RANKS.json")).ok(),
                    categories: std::fs::read_to_string(dir.join("CATEGORIES.json")).ok(),
                })
            }
            Source::Remote { .. } => {
                let orbits = parse_orbits_bin(&self.fetch_bytes("orbits.bin")?)?;
                Ok(Loaded {
                    orbits,
                    sorted_sats: self.fetch_text("sorted_sats.json").ok(),
                    elset: self.fetch_text("elset_min.json").ok(),
                    transmitters: self.fetch_text("transmitters.json").ok(),
                    ranks: self.fetch_text("ranks.json").ok(),
                    categories: self.fetch_text("categories.json").ok(),
                })
            }
        }
    }

    /// Local propagation: the raw element sets (SORTED_SATS.json, true epochs) plus the same side files.
    /// `orbits` comes back empty; the viewer integrates them itself. Files: read from the data folder.
    /// Remote: the server's elsets.json (an older server without it makes this fail, and the caller falls
    /// back to `load`).
    pub fn load_elsets(&self) -> Result<(ElSetMatrix, Loaded), String> {
        let (text, rest) = match self {
            Source::Files { dir, .. } => {
                let p = dir.join("SORTED_SATS.json");
                let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                (text, Loaded {
                    orbits: Vec::new(),
                    sorted_sats: None,
                    elset: std::fs::read_to_string(dir.join("ELSET.json")).ok(),
                    transmitters: std::fs::read_to_string(dir.join("NORADs.json")).ok(),
                    ranks: std::fs::read_to_string(dir.join("SATELLITE_RANKS.json")).ok(),
                    categories: std::fs::read_to_string(dir.join("CATEGORIES.json")).ok(),
                })
            }
            Source::Remote { .. } => {
                let text = self.fetch_text("elsets.json")?;
                (text, Loaded {
                    orbits: Vec::new(),
                    sorted_sats: None,
                    elset: self.fetch_text("elset_min.json").ok(),
                    transmitters: self.fetch_text("transmitters.json").ok(),
                    ranks: self.fetch_text("ranks.json").ok(),
                    categories: self.fetch_text("categories.json").ok(),
                })
            }
        };
        let coe: ElSetMatrix = serde_json::from_str(&text).map_err(|e| format!("element sets: {e}"))?;
        if coe.ncols() == 0 { return Err("element sets: no satellites".into()); }
        Ok((coe, Loaded { sorted_sats: Some(text), ..rest }))
    }

    /// TV settings served by the PC (tv/viewer-tv.toml); None on the desktop or when the PC is not reachable
    pub fn fetch_config(&self) -> Option<String> {
        match self {
            Source::Remote { .. } => self.fetch_text("viewer.toml").ok(),
            _ => None,
        }
    }

    /// Tell Perigee which sky window to rank for. Desktop: write VIEW_REGION.json next to the data.
    /// TV: POST it to the cast server, which writes the same file and re-ranks.
    pub fn send_region(&self, json: &str) -> Result<(), String> {
        match self {
            Source::Files { dir, .. } => std::fs::write(dir.join("VIEW_REGION.json"), json).map_err(|e| e.to_string()),
            Source::Remote { base, token, .. } => {
                let url = format!("{base}/{token}/data/view_region");
                self.agent()?.post(&url).set("Content-Type", "application/json").send_string(json).map(|_| ()).map_err(|e| e.to_string())
            }
        }
    }

    /// Latest rankings text: file read on desktop, HTTPS fetch on the TV
    pub fn fetch_ranks(&self) -> Option<String> {
        match self {
            Source::Files { dir, .. } => std::fs::read_to_string(dir.join("SATELLITE_RANKS.json")).ok(),
            Source::Remote { .. } => self.fetch_text("ranks.json").ok(),
        }
    }

    fn agent(&self) -> Result<ureq::Agent, String> {
        let Source::Remote { cert_sha256, .. } = self else { return Err("not remote".into()) };
        let provider = rustls::crypto::ring::default_provider();
        let verifier = Arc::new(Pinned { sha256: cert_sha256.clone(), provider: provider.clone() });
        let cfg = rustls::ClientConfig::builder_with_provider(Arc::new(provider))
            .with_safe_default_protocol_versions().map_err(|e| e.to_string())?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
        Ok(ureq::builder()
            .tls_config(Arc::new(cfg))
            .timeout(std::time::Duration::from_secs(60))
            .build())
    }

    fn fetch_bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        let Source::Remote { base, token, .. } = self else { return Err("not remote".into()) };
        let url = format!("{base}/{token}/data/{name}");
        let resp = self.agent()?.get(&url).call().map_err(|e| format!("{name}: {e}"))?;
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf).map_err(|e| format!("{name}: {e}"))?;
        Ok(buf)
    }

    fn fetch_text(&self, name: &str) -> Result<String, String> {
        String::from_utf8(self.fetch_bytes(name)?).map_err(|e| format!("{name}: {e}"))
    }
}

//Accept exactly one certificate: the one whose SHA-256 was baked in at build time. Nothing else, no CA.
#[derive(Debug)]
struct Pinned { sha256: Vec<u8>, provider: rustls::crypto::CryptoProvider }

impl rustls::client::danger::ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        use sha2::Digest;
        let digest = sha2::Sha256::digest(end_entity.as_ref());
        if digest.as_slice() == self.sha256.as_slice() {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General("server certificate does not match the pinned fingerprint".into()))
        }
    }
    fn verify_tls12_signature(&self, message: &[u8], cert: &rustls::pki_types::CertificateDer<'_>, dss: &rustls::DigitallySignedStruct)
        -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }
    fn verify_tls13_signature(&self, message: &[u8], cert: &rustls::pki_types::CertificateDer<'_>, dss: &rustls::DigitallySignedStruct)
        -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

/// orbits.bin written by tv/server.py:
///   "PGO1"  u32 n_sats  f64 step_seconds
///   per sat: u32 norad  f64 epoch_jd  u32 ncols  then ncols x 6 f32 (x y z vx vy vz), column by column
fn parse_orbits_bin(b: &[u8]) -> Result<Vec<Matrix6xX<f64>>, String> {
    let mut p = 0usize;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        if *p + n > b.len() { return Err("orbits.bin truncated".into()); }
        let s = &b[*p..*p + n]; *p += n; Ok(s)
    };
    if take(&mut p, 4)? != b"PGO1" { return Err("orbits.bin: bad magic".into()); }
    let n = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
    let _step = f64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let _norad = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap());
        let _epoch = f64::from_le_bytes(take(&mut p, 8)?.try_into().unwrap());
        let ncols = u32::from_le_bytes(take(&mut p, 4)?.try_into().unwrap()) as usize;
        let raw = take(&mut p, ncols * 6 * 4)?;
        let mut data = Vec::with_capacity(ncols * 6);
        for k in 0..ncols * 6 {
            data.push(f32::from_le_bytes(raw[k * 4..k * 4 + 4].try_into().unwrap()) as f64);
        }
        out.push(Matrix6xX::from_vec(data));
    }
    Ok(out)
}

pub fn hex_to_bytes(s: &str) -> Vec<u8> {
    let s = s.trim();
    (0..s.len() / 2).filter_map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()).collect()
}
