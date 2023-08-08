use ndarray::{Array1, Array3, OwnedRepr, Ix3, Array};
use ndarray_interp::{interp2d::{Interp2D, Biliniar}, Interp2DVec};
use thiserror::Error;

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
    lut: Option<Array3<f64>>,
    alpha: Option<Array1<f64>>,
    re: Option<Array1<f64>>,
    symmetric: bool,
}

impl AerofoilBuilder {
    pub fn new() -> Self {
        AerofoilBuilder { lut: None, alpha: None, re: None, symmetric: true }
    }

    pub fn data(mut self, cl_cd: Array3<f64>) -> Self {
        self.lut = Some(cl_cd);
        self
    }

    pub fn alpha(mut self, alpha: Array1<f64>) -> Self{
        self.alpha = Some(alpha);
        self
    }

    pub fn re(mut self, re: Array1<f64>) -> Self {
        self.re = Some(re);
        self
    }

    pub fn symmetric(mut self, yes: bool) -> Self{
        self.symmetric = yes;
        self
    }

    pub fn build(self) -> Result<Aerofoil, AerofoilBuildError>{
        let AerofoilBuilder { lut, alpha, re, symmetric } = self;
        let lut = match lut {
            Some(lut) => lut,
            None => return Err(AerofoilBuildError::MissingData),
        }; 
        let alpha = match alpha {
            Some(alpha) => alpha,
            None => return Err(AerofoilBuildError::MissingAlpha),
        };
        let re = match re {
            Some(re) => re,
            None => return Err(AerofoilBuildError::MissingRe),
        };

        if !lut.shape()[2] == 2 {

        };

        todo!()
    }
}

#[derive(Debug, Error)]
pub enum AerofoilBuildError {
    #[error("No aerofil data was provided")]
    MissingData,
    #[error("No angle of attac data was provided")]
    MissingAlpha,
    #[error("No raynolds nurbers was provided")]
    MissingRe,
    #[error("")]
    WrongDataShape,
}