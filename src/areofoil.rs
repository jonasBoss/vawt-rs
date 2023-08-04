use ndarray::{Array1, Array3, OwnedRepr, Ix3};
use ndarray_interp::{interp2d::{Interp2D, Biliniar}, Interp2DVec};

#[derive(Debug)]
/// Coefficient of lift and drag
pub struct ClCd([f64; 2]);
impl AsRef<[f64; 2]> for ClCd {
    fn as_ref(&self) -> &[f64; 2] {
        &self.0
    }
}


/// An Aerofoil implemented using a LUT.
#[derive(Debug)]
pub struct Aerofoil {
    lut: Interp2DVec<f64, Biliniar>
}

impl Aerofoil {
    pub fn builder() -> AerofoilBuilder {
        AerofoilBuilder::new()
    }

    /// The coefficient of lift and drag ([`ClCd`]) at the given Reynolds number and angle of attack
    pub fn cl_cd(&self, alpha: f64, re: f64) -> ClCd{
        let clcd = self.lut.interp(alpha, re).unwrap();
        assert!(clcd.len() == 2);
        ClCd([clcd[0], clcd[1]])
    }
}

#[derive(Debug)]
pub struct AerofoilBuilder{
    /// Look up tabel for [cl, cd] over Re, alpha
    lut: Vec<Vec<[f64;2]>>,
    alpha: Vec<f64>,
    re: Vec<f64>,
}

impl AerofoilBuilder {
    pub fn new() -> Self {
        AerofoilBuilder { lut: Vec::new(), alpha: Vec::new(), re: Vec::new() }
    } 
    
    pub fn build() -> Aerofoil{
        todo!()
    }
}