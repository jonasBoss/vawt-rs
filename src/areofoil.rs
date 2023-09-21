use std::f64::consts::PI;

use itertools::Itertools;
use ndarray::{s, stack, Array, Array2, ArrayView1, ArrayViewMut1, Axis};
use ndarray_interp::{
    interp1d::{Interp1D, Linear},
    interp2d::{Biliniar, Interp2D, Interp2DVec},
    vector_extensions::{Monotonic, VectorExtensions},
};
use thiserror::Error;

use crate::rot_vec;

#[derive(Debug)]
/// Coefficient of lift and drag
pub struct ClCd(f64, f64);

impl ClCd {
    /// convert lift and drag coeffitients to normal and tangential coefficients
    ///
    /// # returns
    /// `(c_n: f64, c_t: f64)`
    pub fn to_tangential(&self, alpha: f64, beta: f64) -> (f64, f64) {
        rot_vec(self.0, -self.1, alpha + beta)
    }

    pub fn to_global(&self, alpha: f64, beta: f64, theta: f64) -> (f64, f64) {
        rot_vec(self.0, -self.1, alpha + beta + theta)
    }

    pub fn cl(&self) -> f64 {
        self.0
    }

    pub fn cd(&self) -> f64 {
        self.1
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

    /// The coefficient of lift and drag ([`ClCd`]) at the given Reynolds number
    /// and angle of attack in radians
    pub fn cl_cd(&self, alpha: f64, re: f64) -> ClCd {
        if self.symmetric {
            let mut clcd = self.lut.interp(alpha.abs(), re).unwrap();
            clcd[0] *= alpha.signum();
            ClCd(clcd[0], clcd[1])
        } else {
            let clcd = self.lut.interp(alpha, re).unwrap();
            ClCd(clcd[0], clcd[1])
        }
    }
}

#[derive(Debug)]
pub struct AerofoilBuilder {
    /// data rows for [alpha, cl, cd] over Re
    /// where alpha is given in radians
    data: Vec<Array2<f64>>,
    /// reynolds number corresponding to each data row
    re: Vec<f64>,
    /// is the data for a symmetric profile?
    symmetric: bool,
    /// the aspect ratio of the aerofoil
    aspect_ratio: f64,
    /// is the data given for infinite wings and needs do be
    /// convertet according to the aspect ratio
    update_aspect_ratio: bool,
}

impl AerofoilBuilder {
    pub fn new() -> Self {
        AerofoilBuilder {
            data: Vec::new(),
            re: Vec::new(),
            symmetric: false,
            aspect_ratio: f64::INFINITY,
            update_aspect_ratio: false,
        }
    }

    /// add profile data as an 2-d array of `[[alpha_1, cl_1, cd_1], [alpha_2, cl_2, cd_2], ..]`,
    /// where alpha is given in radians.
    /// The data must be strict monotonic rising over alpha, and the reynolds number has to be uniqe
    pub fn add_data_row(
        &mut self,
        data: Array2<f64>,
        re: f64,
    ) -> Result<&mut Self, AerofoilBuildError> {
        if self.re.contains(&re) {
            return Err(AerofoilBuildError::Duplicate(re));
        };
        if data.shape()[1] != 3 {
            return Err(AerofoilBuildError::ShapeError(
                "data must contain 3 elements in the second dimension".into(),
            ));
        };
        if !matches!(
            data.index_axis(Axis(1), 0).monotonic_prop(),
            Monotonic::Rising { strict: true }
        ) {
            return Err(AerofoilBuildError::Monotonic(
                "alpha values must be strict monotonic rising".into(),
            ));
        };
        self.data.push(data);
        self.re.push(re);
        Ok(self)
    }

    /// When this is set, only data for positive angles of attack need to be provided.
    pub fn symmetric(&mut self, yes: bool) -> &mut Self {
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
    pub fn update_aspect_ratio(&mut self, yes: bool) -> &mut Self {
        self.update_aspect_ratio = yes;
        self
    }

    /// set the aspect ratio of the areofoil.
    ///
    /// When the data does not reflect this aspect ratio,
    /// but instead is profile data for an infinte aspect ratio set [`update_aspect_ratio`]
    pub fn set_aspect_ratio(&mut self, ar: f64) -> &mut Self {
        self.aspect_ratio = ar;
        self
    }

    pub fn build(&self) -> Result<Aerofoil, AerofoilBuildError> {
        let data = if self.update_aspect_ratio {
            self.transformed_data_rows()
        } else {
            self.data.clone()
        };

        // the order of re is not guaranteed, sort it accoridngly
        let (re, data): (Vec<_>, Vec<_>) = self
            .re
            .iter()
            .copied()
            .zip(data)
            .sorted_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unzip();

        // the alpha values for the datarows may not agree, so we need create a new alpha axis which contains all unique values
        let mut resampled_alpha: Vec<f64> = data
            .iter()
            .flat_map(|arr| arr.slice(s![.., 0]).into_iter().copied())
            .collect();
        resampled_alpha.sort_by(|a, b| a.partial_cmp(b).unwrap());
        resampled_alpha.dedup_by(|a, b| approx::abs_diff_eq!(a, b, epsilon = f64::EPSILON));
        let resampled_alpha = Array::from(resampled_alpha);

        // resample the data rows with the new alpha axis by interpolating linearly. Also drop the old alpha row from each datarow
        let resampled_data: Vec<_> = data
            .iter()
            .map(|data_row| {
                let clcd = data_row.slice(s![.., 1..]);
                let alpha = data_row.index_axis(Axis(1), 0);
                Interp1D::builder(clcd)
                    .x(alpha)
                    .strategy(Linear::new().extrapolate(true))
                    .build()
                    .unwrap()
                    .interp_array(&resampled_alpha)
                    .unwrap()
            })
            .collect();

        // stack the datarows into a 3d array with alpha as the first axis, re as the second and [cl, cd] as the third
        let data = stack(
            Axis(1),
            &resampled_data.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        let lut = Interp2D::builder(data)
            .x(resampled_alpha)
            .y(Array::from(re))
            .strategy(Biliniar::new().extrapolate(true))
            .build()?;
        Ok(Aerofoil::new(lut, self.symmetric))
    }

    fn transformed_data_rows(&self) -> Vec<Array2<f64>> {
        if !self.update_aspect_ratio {
            return self.data.clone();
        }

        if self.aspect_ratio >= 98.0 {
            return self.data.clone();
        }

        self.data
            .iter()
            .cloned()
            .map(|data_row| self.transform_row(data_row))
            .collect()
    }

    fn transform_row(&self, mut data_row: Array2<f64>) -> Array2<f64> {
        let zero_idx = data_row.index_axis(Axis(0), 0).get_lower_index(0.0);
        let mut stall_idx = None;
        let mut last_cl = data_row[(zero_idx, 1)];
        for (idx, mut datapoint) in data_row
            .slice_mut(s![(zero_idx + 1).., ..])
            .axis_iter_mut(Axis(0))
            .enumerate()
        {
            if last_cl > datapoint.cl() {
                stall_idx = Some(idx + zero_idx);
                break; // cl has dropped (stall)
            }
            last_cl = datapoint.cl();
            datapoint.lanchester_prandtl(self.aspect_ratio);
        }
        let stall_idx = stall_idx.unwrap_or_else(|| panic!("stall point not found!"));
        let stall = data_row.index_axis(Axis(0), stall_idx).into_owned();

        // above the stall point we calculate the data for each degree up to 90
        let new_len = stall_idx + 90 + 1 - stall[0].to_degrees().floor() as usize;
        let mut new_data = Array::zeros((new_len, 3));
        new_data
            .slice_mut(s![..(stall_idx + 1), ..])
            .assign(&data_row.slice(s![..(stall_idx + 1), ..]));
        new_data
            .slice_mut(s![(stall_idx + 1).., 0])
            .assign(&Array::linspace(
                stall[0].to_degrees().ceil().to_radians(),
                PI / 2.0,
                new_len - stall_idx - 1,
            ));
        data_row = new_data;

        for mut datapoint in data_row
            .slice_mut(s![(stall_idx + 1).., ..])
            .axis_iter_mut(Axis(0))
        {
            datapoint.viterna_corrigan(stall.view(), self.aspect_ratio);
        }

        if self.symmetric {
            return data_row;
        }
        todo!("updating the AR for asymmetric profiles is not yet implemented");
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

trait DataPoint {
    fn a(&self) -> f64;
    fn cl(&self) -> f64;
    fn cd(&self) -> f64;
    fn lanchester_prandtl(&mut self, aspect_ratio: f64);
    fn viterna_corrigan(&mut self, stall: ArrayView1<f64>, aspect_ratio: f64);
}

impl<'a> DataPoint for ArrayViewMut1<'a, f64> {
    fn a(&self) -> f64 {
        self[0]
    }

    fn cl(&self) -> f64 {
        self[1]
    }

    fn cd(&self) -> f64 {
        self[2]
    }

    fn lanchester_prandtl(&mut self, aspect_ratio: f64) {
        self[2] = self.cd() + self.cl().powi(2) / (PI * aspect_ratio);
        self[0] = self.a() + self.cl() / (PI * aspect_ratio);
    }

    fn viterna_corrigan(&mut self, stall: ArrayView1<f64>, aspect_ratio: f64) {
        let cd_max = if aspect_ratio > 50.0 {
            2.01
        } else {
            1.1 + 0.018 * aspect_ratio
        };
        let kd = (stall.cd() - cd_max * stall.a().sin().powi(2)) / stall.a().cos();
        let kl = (stall.cl() - cd_max * stall.a().sin() * stall.a().cos()) * stall.a().sin()
            / stall.a().cos().powi(2);
        self[1] =
            cd_max / 2.0 * (2.0 * self.a()).sin() + kl * self.a().cos().powi(2) / self.a().sin();
        self[2] = cd_max * self.a().sin().powi(2) + kd * self.a().cos();
    }
}

impl<'a> DataPoint for ArrayView1<'a, f64> {
    fn a(&self) -> f64 {
        self[0]
    }

    fn cl(&self) -> f64 {
        self[1]
    }

    fn cd(&self) -> f64 {
        self[2]
    }

    fn lanchester_prandtl(&mut self, _aspect_ratio: f64) {
        todo!("not possible")
    }

    fn viterna_corrigan(&mut self, _stall: ArrayView1<f64>, _aspect_ratio: f64) {
        todo!("not possible")
    }
}
