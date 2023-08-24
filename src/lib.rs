use std::f64::consts::PI;

use areofoil::Aerofoil;
use ndarray::{array, concatenate, s, Array, Array1, Array2, Axis, Zip};
use ndarray_interp::interp1d::{Interp1D, Linear};
use streamtube::StreamTube;

pub mod areofoil;
pub mod streamtube;

/// 2d rotation matrix for angle phi (in radians)
fn rot_mat(phi: f64) -> Array2<f64> {
    array![[phi.cos(), (-phi).sin()], [phi.sin(), phi.cos()],]
}

#[derive(Debug)]
pub struct VAWTSolver<'a> {
    aerofoil: &'a Aerofoil,
    n_streamtubes: usize,
    tsr: f64,
    re: f64,
    solidity: f64,
    epsilon: f64,
    verbosity: Verbosity,
}

/// A VAWT case and solver settings
impl<'a> VAWTSolver<'a> {
    /// create a new Solver with the following default values:
    ///
    /// - `n_streamtubes = 50` Number of streamtubes over the whole turbine
    /// - `tsr = 2.0` Tipspeed ratio of the turbine
    /// - `re = 60_000.0` Reynolds number of the turbine
    /// - `solidity = 0.1` Solidity of the Turbine
    /// - `epsilon = 0.01` the solution accuracy for a
    pub fn new(aerofoil: &'a Aerofoil) -> VAWTSolver<'a> {
        VAWTSolver {
            aerofoil,
            n_streamtubes: 50,
            tsr: 2.0,
            re: 60_000.0,
            solidity: 0.1,
            epsilon: 0.01,
            verbosity: Verbosity::Normal,
        }
    }

    /// update the number of streamtubes for the solution if n is not a multiple of 2 `n+1` is used.
    pub fn n_streamtubes(&mut self, n: usize) -> &mut Self {
        let n = if n % 2 == 0 { n } else { n + 1 };
        self.n_streamtubes = n;
        self
    }

    /// update the tipspeed ratio for the solution
    pub fn tsr(&mut self, tsr: f64) -> &mut Self {
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

    pub fn verbosity(&mut self, verbosity: Verbosity) -> &mut Self {
        self.verbosity = verbosity;
        self
    }

    /// solve the VAWT Turbine with a constant beta angle in radians
    pub fn solve_with_beta(&self, beta: f64) -> VAWTSolution<'a> {
        let d_t_half = PI / self.n_streamtubes as f64;
        let theta = Array::linspace(d_t_half, PI * 2.0 - d_t_half, self.n_streamtubes);
        let beta = Array::from_elem(self.n_streamtubes, beta);
        let mut a_0 = Array::zeros(self.n_streamtubes);
        let mut a_up: Array1<f64> = Array::zeros(self.n_streamtubes / 2);
        let mut a_down: Array1<f64> = Array::zeros(self.n_streamtubes / 2);

        let slice_up = s![..self.n_streamtubes / 2];
        let slice_down = s![self.n_streamtubes / 2 ..;-1];

        let VAWTSolver {
            aerofoil,
            n_streamtubes,
            tsr,
            re,
            solidity,
            epsilon,
            verbosity,
        } = *self;
        let turbine = Turbine {
            re,
            tsr,
            solidity,
            aerofoil,
        };

        Zip::from(theta.slice(slice_up))
            .and(theta.slice(slice_down))
            .and(beta.slice(slice_up))
            .and(beta.slice(slice_down))
            .and(a_up.view_mut())
            .and(a_down.slice_mut(s![..;-1]))
            .for_each(
                |&theta_up, &theta_down, &beta_up, &beta_down, a_up, a_down| {
                    verbosity.print(format!("solving for theta = {}°", theta_up.to_degrees()));
                    *a_up = StreamTube::new(theta_up, beta_up, 0.0).solve_a(&turbine, epsilon);

                    verbosity.print(format!("solving for theta = {}°", theta_down.to_degrees()));
                    *a_down =
                        StreamTube::new(theta_down, beta_down, *a_up).solve_a(&turbine, epsilon);
                },
            );

        a_0.slice_mut(slice_down).assign(&a_up);
        let a = concatenate![Axis(0), a_up, a_down];
        VAWTSolution {
            turbine,
            n_streamtubes,
            theta,
            beta,
            a,
            a_0,
            epsilon,
        }
    }

    /// solve the VAWT Turbine with a provided beta angle as function of theta in radians
    pub fn solve_with_beta_fn(&self, _beta: impl Fn(f64) -> f64) -> VAWTSolution<'a> {
        todo!()
    }

    /// solve the VAWT Turbine while optimizing beta
    pub fn solve_optimize_beta(&self) -> VAWTSolution<'a> {
        todo!()
    }
}

/// Turbine settings for the VAWT case
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

/// The solution of a VAWT case
#[derive(Debug)]
pub struct VAWTSolution<'a> {
    turbine: Turbine<'a>,
    n_streamtubes: usize,
    theta: Array1<f64>,
    beta: Array1<f64>,
    a: Array1<f64>,
    a_0: Array1<f64>,
    epsilon: f64,
}

impl<'a> VAWTSolution<'a> {
    /// Torque and power coefficient of the turbine
    pub fn ct_cp(&self) -> (f64, f64) {
        let mut ct = Zip::from(&self.theta)
            .and(&self.beta)
            .and(&self.a)
            .and(&self.a_0)
            .fold(0.0, |acc, &theta, &beta, &a, &a_0| {
                let tube = StreamTube::new(theta, beta, a_0);
                let (ct, _) = tube.c_tan_cf_tan(a, &self.turbine);
                let (w, ..) = tube.w_alpha_re(a, &self.turbine);
                acc + ct * w.powi(2)
            });
        ct = ct * self.turbine.solidity / (self.n_streamtubes as f64);
        let cp = ct * self.turbine.tsr;
        (ct, cp)
    }

    pub fn beta(&self, theta: f64) -> f64 {
        Interp1D::new_unchecked(
            self.theta.view(),
            self.beta.view(),
            Linear::new().extrapolate(true),
        )
        .interp_scalar(theta)
        .unwrap()
    }

    pub fn a(&self, theta: f64) -> f64 {
        Interp1D::new_unchecked(
            self.theta.view(),
            self.a.view(),
            Linear::new().extrapolate(true),
        )
        .interp_scalar(theta)
        .unwrap()
    }

    pub fn a_0(&self, theta: f64) -> f64 {
        Interp1D::new_unchecked(
            self.theta.view(),
            self.a_0.view(),
            Linear::new().extrapolate(true),
        )
        .interp_scalar(theta)
        .unwrap()
    }

    pub fn thrust_error(&self, theta: f64) -> f64 {
        let a = self.a(theta);
        self.streamtube(theta).thrust_error(a, &self.turbine)
    }

    pub fn c_tan(&self, theta: f64) -> f64 {
        let a = self.a(theta);
        let (ct, _) = self.streamtube(theta).c_tan_cf_tan(a, &self.turbine);
        ct
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn w_alpha_re(&self, theta: f64) -> (f64, f64, f64) {
        self.streamtube(theta)
            .w_alpha_re(self.a(theta), &self.turbine)
    }

    fn streamtube(&self, theta: f64) -> StreamTube {
        let a_0 = self.a_0(theta);
        let beta = self.beta(theta);
        StreamTube::new(theta, beta, a_0)
    }
}

///
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Verbosity {
    Silent,
    Normal,
    Debug,
}

impl Verbosity {
    fn print<S: AsRef<str>>(&self, s: S) {
        use Verbosity::*;
        if *self == Normal || *self == Debug {
            println!("{}", s.as_ref())
        }
    }

    fn debug<S: AsRef<str>>(&self, s: S) {
        use Verbosity::*;
        if *self == Debug {
            println!("{}", s.as_ref())
        }
    }
}
