use std::f64::consts::PI;

use ndarray::{s, stack, Array, Array1, Array2, Array3, Axis, Dim, Ix3, OwnedRepr, Slice, ArrayViewMut1, ArrayBase, ArrayView1};
use ndarray_interp::{
    interp1d::{Interp1D, Linear},
    interp2d::{Biliniar, Interp2D, Interp2DVec}, vector_extensions::{VectorExtensions, Monotonic},
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

    /// add profile data as an 2-d array of `[[alpha_1, cl_1, cd_1], [alpha_2, cl_2, cd_2], ..]`
    /// the data must be strict monotonic rising over alpha, and the reynolds number must be new
    pub fn add_data_row(&mut self, data: Array2<f64>, re: f64) -> Result<&mut Self, AerofoilBuildError> {
        if self.re.contains(&re){return Err(AerofoilBuildError::Duplicate(re));};
        if data.shape()[1] != 3 {return Err(AerofoilBuildError::ShapeError("data must contain 3 elements in the second dimension".into()));};
        if !matches!(data.index_axis(Axis(0), 0).monotonic_prop(), Monotonic::Rising { strict: true }) {return Err(AerofoilBuildError::Monotonic("alpha values must be strict monotonic rising".into()));};
        self.data.push(data);
        self.re.push(re);
        Ok(self)
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
        for data in self.data.iter_mut(){
            let zero_idx = data.index_axis(Axis(0), 0).get_lower_index(0.0);
            let mut stall_idx = None;
            let mut last_cl = data[(1,zero_idx)];
            for (idx, mut datapoint) in data.slice_axis_mut(Axis(0), Slice::new((zero_idx+1) as isize, None, 1)).axis_iter_mut(Axis(0)).enumerate() {
                if last_cl > datapoint.cl() {
                    stall_idx = Some(idx + zero_idx);
                    break; // cl has dropped (stall)
                }
                last_cl = datapoint.cl();
                datapoint.lanchester_prandtl(self.aspect_ratio);
            }
            let stall_idx = stall_idx.unwrap_or_else(||panic!("stall point not found!"));
            let stall = data.index_axis(Axis(0), stall_idx).into_owned();
            for mut datapoint in data.slice_axis_mut(Axis(0), Slice::new((stall_idx+1) as isize, None, 1)).axis_iter_mut(Axis(0)) {
                datapoint.viterna_corrigan(stall.view(), self.aspect_ratio);
            }
            todo!("negative alpha")
        }
    }
}

#[derive(Debug, Error)]
pub enum AerofoilBuildError {
    #[error("{0}")]
    ShapeError(String),
    #[error("{0}")]
    NotEnoughtData(String),
    #[error("{0}")]
    Monotonic(String),
    #[error("data for the reynolds-number {0} is already stored")]
    Duplicate(f64),
}

impl From<ndarray_interp::BuilderError> for AerofoilBuildError {
    fn from(value: ndarray_interp::BuilderError) -> Self {
        match value {
            ndarray_interp::BuilderError::NotEnoughData(s) => AerofoilBuildError::NotEnoughtData(s),
            ndarray_interp::BuilderError::Monotonic(_) => unreachable!(),
            ndarray_interp::BuilderError::AxisLenght(_) => unreachable!(),
            ndarray_interp::BuilderError::DimensionError(_) => unreachable!(),
        }
    }
}


trait DataPoint{
    fn a(&self) -> f64;
    fn cl(&self) -> f64;
    fn cd(&self) -> f64;
    fn lanchester_prandtl(&mut self, aspect_ratio: f64);
    fn viterna_corrigan(&mut self, stall: ArrayView1<f64>, aspect_ratio: f64);
}

impl<'a> DataPoint for ArrayViewMut1<'a, f64>{
    fn a(&self) -> f64 {
        self[0]
    }

    fn cl(&self) -> f64 {
        self[1]
    }

    fn cd(&self) -> f64 {
        self[2]
    }

    fn lanchester_prandtl(&mut self, aspect_ratio: f64){
        self[2] = self.cd() + self.cl().powi(2) /(PI * aspect_ratio);
        self[0] = self.a() + self.cl() / (PI * aspect_ratio);
    }

    fn viterna_corrigan(&mut self, stall: ArrayView1<f64>, aspect_ratio: f64) {
        let cd_max = if aspect_ratio > 50.0 {2.01} else {1.1+0.018*aspect_ratio};
        let kd = (stall.cd() - cd_max * stall.a().sin().powi(2))/stall.a().cos();
        let kl = (stall.cl() - cd_max * stall.a().sin() * stall.a().cos()) * stall.a().sin() / stall.a().cos().powi(2);
        self[1] = cd_max/2.0* (2.0*self.a()).sin() + kl * self.a().cos().powi(2)/self.a().sin();
        self[2] = cd_max * self.a().sin().powi(2) + kd * self.a().cos();
    }
}

impl<'a> DataPoint for ArrayView1<'a, f64>{
    fn a(&self) -> f64 {
        self[0]
    }

    fn cl(&self) -> f64 {
        self[1]
    }

    fn cd(&self) -> f64 {
        self[2]
    }

    fn lanchester_prandtl(&mut self, aspect_ratio: f64){
        todo!("not possible")
    }

    fn viterna_corrigan(&mut self, stall: ArrayView1<f64>, aspect_ratio: f64) {
        todo!("not possible")
    }
}