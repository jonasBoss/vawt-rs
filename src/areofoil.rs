use ndarray::{Array1, Array3};

#[derive(Debug)]
/// Coefficient of lift and drag
pub struct ClCd([f64; 2]);
impl AsRef<[f64; 2]> for ClCd {
    fn as_ref(&self) -> &[f64; 2] {
        &self.0
    }
}

/// The main aerofoil trait used by VAWT
pub trait Aerofoil {
    /// The coefficient of lift and drag ([`ClCd`]) at the given Reynolds number and angle of attack
    fn cl_cd(&self, re: f64, alpha: f64) -> ClCd;
}

/// An Aerofoil implemented using a LUT.
pub struct LutAerofoil {
    /// Look up tabel for [cl, cd] over Re, alpha
    lut: Array3<f64>,

    /// alpha idx
    alpha_idx: Array1<f64>,

    /// Re idx
    re_idx: Array1<f64>,
}

impl LutAerofoil {}

impl Aerofoil for LutAerofoil {
    fn cl_cd(&self, re: f64, alpha: f64) -> ClCd {
        todo!()
    }
}
