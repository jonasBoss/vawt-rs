use ndarray::{s, stack, Array, Array1, Array2, Array3, Axis, Dim, Ix3, OwnedRepr};
use ndarray_interp::{
    interp1d::{Interp1D, Linear},
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
    pub fn builder() -> AerofoilBuilder {
        AerofoilBuilder::new()
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
    /// data rows for [alpha, cl, cd] over Re
    data: Vec<Array2<f64>>,
    re: Vec<f64>,
    symmetric: bool,
    aspect_ratio: f64,
    update_aspect_ratio: bool,
}

impl AerofoilBuilder {
    pub fn new() -> Self {
        AerofoilBuilder {
            data: Vec::new(),
            re: Vec::new(),
            symmetric: true,
            aspect_ratio: f64::INFINITY,
            update_aspect_ratio: false,
        }
    }

    pub fn add_data_row(mut self, data: Array2<f64>, re: f64) -> Self {
        assert!(!self.re.contains(&re));
        self.data.push(data);
        self.re.push(re);
        self
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
    ///
    /// # Note
    /// This is currently only for symmetric profiles implemented.
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

    pub fn build(mut self) -> Result<Aerofoil, AerofoilBuildError> {
        if self.update_aspect_ratio {
            self.change_ar()
        }

        let AerofoilBuilder {
            mut data,
            re,
            symmetric,
            aspect_ratio,
            update_aspect_ratio,
        } = self;

        // the order of re is not guaranteed, sort it accoridngly
        let mut data = re.into_iter().zip(data).collect::<Vec<_>>();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (re, mut data) = data.into_iter().fold(
            (Vec::new(), Vec::new()),
            |(mut re_acc, mut d_acc), (re, d)| {
                re_acc.push(re);
                d_acc.push(d);
                (re_acc, d_acc)
            },
        );

        // resample the data so that we get a grid whith all unique alpha values
        let mut alpha: Vec<f64> = data
            .iter()
            .flat_map(|arr| arr.slice(s![.., 0]).into_iter().map(|f| *f))
            .collect();
        alpha.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = Array::from_iter(alpha.into_iter().fold(Vec::new(), |mut acc, f| {
            match acc.last() {
                None => acc.push(f),
                Some(&last) => {
                    if last != f {
                        acc.push(f)
                    }
                }
            };
            acc
        }));
        data = data
            .into_iter()
            .map(|arr| {
                let clcd = arr.slice(s![.., 1..]);
                let alpha = arr.index_axis(Axis(1), 0);
                Interp1D::builder(clcd)
                    .x(alpha)
                    .strategy(Linear::new().extrapolate(true))
                    .build()
                    .unwrap()
                    .interp_array(&alpha)
                    .unwrap()
            })
            .collect();
        let data = stack(Axis(1), &*data.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();

        let lut = Interp2D::builder(data)
            .x(alpha)
            .y(Array::from(re))
            .strategy(Biliniar)
            .build()?;
        Ok(Aerofoil::new(lut, symmetric))
    }

    fn change_ar(&mut self) {
        if self.aspect_ratio >= 98.0 {
            return;
        }

        //let mut new_data = Array::zeros(self.lut.raw_dim());
        //let mut new_alpha = Array::zeros(self.alpha.raw_dim());

        //for
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
