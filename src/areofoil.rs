use ndarray::{Array, Array1, Array3, Dim, Ix3, OwnedRepr};
use ndarray_interp::{
    interp2d::{Biliniar, Interp2D},
    Interp2DVec,
};
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
    lut: Interp2DVec<f64, Biliniar>,
    symmetric: bool,
}

impl Aerofoil {
    pub fn builder(data: Array3<f64>, alpha: Array1<f64>, re: Array1<f64>) -> AerofoilBuilder {
        AerofoilBuilder::new(data, alpha, re)
    }

    fn new(data: Interp2DVec<f64, Biliniar>, symmetric: bool) -> Self {
        Aerofoil {
            lut: data,
            symmetric,
        }
    }

    /// The coefficient of lift and drag ([`ClCd`]) at the given Reynolds number and angle of attack
    pub fn cl_cd(&self, alpha: f64, re: f64) -> ClCd {
        if self.symmetric {
            let sign = alpha / alpha.abs();
            let clcd = self.lut.interp(alpha.abs(), re).unwrap();
            ClCd([sign * clcd[0], clcd[1]])
        } else {
            let clcd = self.lut.interp(alpha, re).unwrap();
            ClCd([clcd[0], clcd[1]])
        }
    }
}

#[derive(Debug)]
pub struct AerofoilBuilder {
    /// Look up tabel for [cl, cd] over Re, alpha
    lut: Array3<f64>,
    alpha: Array1<f64>,
    re: Array1<f64>,
    symmetric: bool,
    aspect_ratio: f64,
    update_aspect_ratio: bool,
}

impl AerofoilBuilder {
    pub fn new(data: Array3<f64>, alpha: Array1<f64>, re: Array1<f64>) -> Self {
        AerofoilBuilder {
            lut: data,
            alpha,
            re,
            symmetric: true,
            aspect_ratio: f64::INFINITY,
            update_aspect_ratio: false,
        }
    }

    /// When this is set, only data for positive angles of attack need to be provided.
    pub fn symmetric(mut self, yes: bool) -> Self {
        self.symmetric = yes;
        self
    }

    /// Assume the provided data to be for a infinite aspect ratio.
    /// 
    /// Updtate the data with the Lanchester-Prandtl model below the stalling angle
    /// and with the Viterna-Corrigan above the stall angle
    pub fn update_aspect_ratio(mut self, yes: bool) -> Self {
        self.update_aspect_ratio = yes;
        self
    }

    /// set the aspect ratio of the areofoil. 
    /// 
    /// When the data does not reflect this aspect ratio,
    /// but instead is profile data for an infinte aspect ratio set [`update_aspect_ratio`]
    pub fn set_aspect_ratio(mut self, ar: f64) -> Self {
        self.aspect_ratio = ar;
        self
    }

    pub fn build(self) -> Result<Aerofoil, AerofoilBuildError> {
        let AerofoilBuilder {
            lut,
            alpha,
            re,
            symmetric,
            aspect_ratio,
            update_aspect_ratio,
        } = self;
        let expected = Ix3(alpha.len(), re.len(), 2);
        if !(lut.raw_dim() == expected) {
            return Err(AerofoilBuildError::WrongDataShape(expected, lut.raw_dim()));
        };

        if update_aspect_ratio && (aspect_ratio < 98.0) {

        }


        let lut = Interp2D::builder(lut)
            .x(alpha)
            .y(re)
            .strategy(Biliniar)
            .build()?;
        Ok(Aerofoil::new(lut, symmetric))
    }
}

#[derive(Debug, Error)]
pub enum AerofoilBuildError {
    #[error("data must have shape `[alpha.len(), re.len(), 2]` \n expected: {0:?}, found: {1:?}")]
    WrongDataShape(Ix3, Ix3),
    #[error("{0}")]
    NotEnoughtData(String),
    #[error("{0}")]
    Monotonic(String),
}

impl From<ndarray_interp::BuilderError> for AerofoilBuildError {
    fn from(value: ndarray_interp::BuilderError) -> Self {
        match value {
            ndarray_interp::BuilderError::NotEnoughData(s) => AerofoilBuildError::NotEnoughtData(s),
            ndarray_interp::BuilderError::Monotonic(s) => AerofoilBuildError::Monotonic(s),
            ndarray_interp::BuilderError::AxisLenght(_) => unreachable!(),
            ndarray_interp::BuilderError::DimensionError(_) => unreachable!(),
        }
    }
}
