use std::f64::consts::PI;

use ndarray::{Array1, Ix1, Array, Zip, s, concatenate, Axis};
use ndarray_interp::interp1d::{Interp1D, Interp1DOwned, Interp1DScalar, Linear};

use crate::{areofoil::Aerofoil, turbine, streamtube::{self, StreamTube}};

#[derive(Debug, Clone)]
pub struct Turbine<'a> {
    /// Raynoldsnumber of the turbine
    pub re: f64,
    /// Tipspeed ratio of the turbine
    pub tsr: f64,
    /// Turbine solidity
    pub solidity: f64,
    /// Aerofoil
    pub aerofoil: &'a Aerofoil,
}

#[derive(Debug)]
pub struct  TurbineSolution<'a> {
    turbine: Turbine<'a>,
    n_streamtubes: usize,
    theta: Array1<f64>,
    beta: Array1<f64>,
    a: Array1<f64>,
}

impl<'a> TurbineSolution<'a> {
    pub fn cp(&self) -> f64 {
        todo!()
    }

    pub fn beta(&self, theta: f64) -> f64 {
        Interp1D::new_unchecked(
            self.theta.view(), 
            self.beta.view(), 
            Linear::new().extrapolate(true)
        )
            .interp_scalar(theta)
            .unwrap()
    }

    pub fn a(&self, theta: f64) -> f64 {
        (Interp1D::new_unchecked(
            self.theta.view(), 
            self.a.view(), 
            Linear::new().extrapolate(true)
        ))
            .interp_scalar(theta)
            .unwrap()
    }
 }

#[derive(Debug)]
pub struct VAWTSolver<'a> {
    aerofoil: &'a Aerofoil,
    n_streamtubes: usize,
    tsr: f64,
    re: f64,
    solidity: f64,
    epsilon: f64,
    
}

impl<'a> VAWTSolver<'a> {
    /// create a new Solver with the following default values:
    /// 
    /// - `n_streamtubes = 50` Number of streamtubes over the whole turbine
    /// - `tsr = 2.0` Tipspeed ratio of the turbine
    /// - `re = 60_000.0` Reynolds number of the turbine
    /// - `solidity = 0.1` Solidity of the Turbine
    /// - `epsilon = 0.01` the solution accuracy for a
    pub fn new(aerofoil: &'a Aerofoil) -> Self {
        VAWTSolver {
            aerofoil,
            n_streamtubes: 50,
            tsr: 2.0,
            re: 60_000.0,
            solidity: 0.1,
            epsilon: 0.01,
        }
    }

    /// update the number of streamtubes for the solution if n is not a multiple of 2 `n+1` is used.
    pub fn n_streamtubes(&mut self, n: usize) -> &mut Self {
        let n = if n % 2 == 0 {n} else {n+1};
        self.n_streamtubes = n;
        self
    }

    /// update the tipspeed ratio for the solution
    pub fn tsr(&mut self, tsr:f64) -> &mut Self {
        self.tsr = tsr;
        self
    }

    /// update the raynolds number for the solution
    pub fn re(&mut self, re: f64) -> &mut Self {
        self.re = re;
        self
    }

    /// update the turbine solidity for the solution
    pub fn solidity(&mut self, solidity: f64) -> &mut Self {
        self.solidity = solidity;
        self
    }

    /// solve the VAWT Turbine with a constant beta angle in radians
    pub fn solve_with_beta(&self, beta: f64) -> TurbineSolution<'a> {
        let d_t_half = PI / self.n_streamtubes as f64;
        let theta = Array::linspace(d_t_half, PI * 2.0 - d_t_half, self.n_streamtubes);
        let beta = Array::from_elem(self.n_streamtubes, beta);
        let mut a_up: Array1<f64> = Array::zeros(self.n_streamtubes / 2);
        let mut a_down: Array1<f64> = Array::zeros(self.n_streamtubes / 2);

        let slice_up = s![..self.n_streamtubes / 2];
        let slice_down = s![self.n_streamtubes / 2 ..;-1];

        let VAWTSolver { aerofoil, n_streamtubes, tsr, re, solidity, epsilon} = *self;
        let turbine = Turbine { re, tsr, solidity, aerofoil };

        Zip::from(theta.slice(slice_up))
            .and(theta.slice(slice_down))
            .and(beta.slice(slice_up))
            .and(beta.slice(slice_down))
            .and(a_up.view_mut())
            .and(a_down.slice_mut(s![..;-1]))
            .for_each(|&theta_up, &theta_down, &beta_up, &beta_down, a_up, a_down|
                {
                    println!("solving for theta = {}°", theta_up.to_degrees());
                    *a_up = StreamTube::new(theta_up, beta_up, 0.0).solve_a(&turbine, epsilon);

                    println!("solving for theta = {}°", theta_down.to_degrees());
                    *a_down = StreamTube::new(theta_down, beta_down, *a_up).solve_a(&turbine, epsilon);
                }
            );
        
        let a = concatenate![Axis(0), a_up, a_down];
        TurbineSolution { turbine: turbine, n_streamtubes, theta, beta, a }
    }

    /// solve the VAWT Turbine with a provided beta angle as function of theta in radians
    pub fn solve_with_beta_fn(&self, beta: impl Fn(f64) -> f64) -> TurbineSolution {
        todo!()
    }

    /// solve the VAWT Turbine while optimizing beta
    pub fn solve_optimize_beta(&self) -> TurbineSolution {
        todo!()
    }
}
