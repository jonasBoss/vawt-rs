use std::f64::consts::PI;

use areofoil::Aerofoil;

use argmin::{core::Executor, solver::particleswarm::ParticleSwarm};
use log::info;
use ndarray::{s, Array, Array1, Zip};
use ndarray_interp::interp1d::{Interp1D, Linear};

use streamtube::{OptimizeBeta, StreamTube, StreamTubeSolution};

pub mod areofoil;
pub mod streamtube;

fn rot_vec(x: f64, y: f64, alpha: f64) -> (f64, f64) {
    (
        alpha.cos() * x + (-alpha).sin() * y,
        alpha.sin() * x + alpha.cos() * y,
    )
}

#[derive(Debug)]
pub struct VAWTSolver<'a> {
    aerofoil: &'a Aerofoil,
    n_streamtubes: usize,
    tsr: f64,
    re: f64,
    solidity: f64,
    epsilon: f64,
    particles: usize,
    iterations: u64,
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
    /// - `particles = 8` the number of particles for beta optimization
    /// - `iterations = 30` the number of iterations for beta optimization
    pub fn new(aerofoil: &'a Aerofoil) -> VAWTSolver<'a> {
        VAWTSolver {
            aerofoil,
            n_streamtubes: 50,
            tsr: 2.0,
            re: 60_000.0,
            solidity: 0.1,
            epsilon: 0.01,
            particles: 8,
            iterations: 30,
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

    pub fn epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// set the number of particles for [`solve_optimize_beta()`](VAWTSolver::solve_optimize_beta)
    pub fn particles(&mut self, particles: usize) -> &mut Self {
        self.particles = particles;
        self
    }

    /// set the number of iterations for [`solve_optimize_beta()`](VAWTSolver::solve_optimize_beta)
    pub fn iterations(&mut self, iters: u64) -> &mut Self {
        self.iterations = iters;
        self
    }

    /// solve the VAWT case with a constant beta angle in radians
    pub fn solve_with_beta(&self, beta: f64) -> VAWTSolution<'a> {
        self.solve_with_beta_fn(|_| beta)
    }

    /// solve the VAWT case with a provided beta angle as function of theta in radians
    pub fn solve_with_beta_fn(&self, beta: impl Fn(f64) -> f64 + Sync) -> VAWTSolution<'a> {
        self.iter_streamtubes(|case, &theta_up, &theta_down| {
            let beta_up = beta(theta_up);
            let beta_down = beta(theta_down);
            let a_up = StreamTube::new(theta_up, beta_up, 0.0).solve_a(case, self.epsilon);
            let a_down = StreamTube::new(theta_down, beta_down, a_up).solve_a(case, self.epsilon);

            (beta_up, beta_down, a_up, a_down)
        })
    }

    /// solve the VAWT case while optimizing beta
    pub fn solve_optimize_beta(&self) -> VAWTSolution<'a> {
        self.iter_streamtubes(|case, &theta_up, &theta_down| {
            let cost = OptimizeBeta::new(case, self.epsilon, theta_up, theta_down);
            let solver = ParticleSwarm::new(
                (
                    Array::from_elem(2, -20f64.to_radians()),
                    Array::from_elem(2, 20f64.to_radians()),
                ),
                self.particles,
            );

            let mut res = Executor::new(cost, solver)
                .configure(|state| state.max_iters(self.iterations))
                .run()
                .unwrap();
            let best_param = res.state.take_best_individual().unwrap().position;

            let (beta_up, beta_down) = (best_param[0], best_param[1]);
            let a_up = StreamTube::new(theta_up, beta_up, 0.0).solve_a(case, self.epsilon);
            let a_down = StreamTube::new(theta_down, beta_down, a_up).solve_a(case, self.epsilon);
            (beta_up, beta_down, a_up, a_down)
        })
    }

    /// the cost function that gets optimized by [`solve_optimize_beta()`](VAWTSolver::solve_optimize_beta)
    /// for each streamtube pair
    ///
    /// the parameters to the cost fn is an [`Array1<f64>`] of length 2. One beta value for the upstream,
    /// and one for the downstream streamtube
    ///
    /// # panics
    /// when `theta` is not between 0 ab PI
    pub fn cost_fn(&'a self, theta: f64) -> impl 'a + Fn(&Array1<f64>) -> f64 {
        assert!(0.0 < theta && theta < PI);
        let theta_down = 2.0 * PI - theta;
        let case = self.get_case();

        move |param: &Array1<f64>| {
            OptimizeBeta::new(&case, self.epsilon, theta, theta_down).cost(param)
        }
    }

    /// solve a single streamtube
    pub fn solve_steamtube(&self, theta: f64, beta: f64, a_0: f64) -> StreamTubeSolution<'a> {
        let case = self.get_case();
        let tube = StreamTube::new(theta, beta, a_0);
        let a = tube.solve_a(&case, self.epsilon);
        StreamTubeSolution::new(case, tube, a)
    }

    /// iterate over all streamtubes, applying `solve_fn`.
    ///
    /// `solve_fn` is called for each pair of up and downstream streamtubes with:
    /// `Fn(case: &VAWTCase, theta_up: &f64, theta_down: &f64) -> (beta_up: f64, beta_down: f64, a_up: f64, a_down: f64)`
    fn iter_streamtubes(
        &self,
        solve_fn: impl Fn(&VAWTCase, &f64, &f64) -> (f64, f64, f64, f64) + Sync,
    ) -> VAWTSolution<'a> {
        let d_t_half = PI / self.n_streamtubes as f64;
        let theta = Array::linspace(d_t_half, PI * 2.0 - d_t_half, self.n_streamtubes);
        let mut beta = Array::zeros(self.n_streamtubes);
        let mut a = Array::zeros(self.n_streamtubes);
        let mut a_0 = Array::zeros(self.n_streamtubes);

        let slice_up = s![..self.n_streamtubes / 2];
        let slice_down = s![self.n_streamtubes / 2 ..;-1];

        let (beta_up, beta_down) = beta.multi_slice_mut((slice_up, slice_down));
        let (a_up, a_down) = a.multi_slice_mut((slice_up, slice_down));

        let case = self.get_case();
        let func = |theta_up: &f64,
                    theta_down: &f64,
                    beta_up: &mut f64,
                    beta_down: &mut f64,
                    a_up: &mut f64,
                    a_down: &mut f64| {
            info!(
                "solving for theta = {}° and theta = {}°",
                theta_up.to_degrees(),
                theta_down.to_degrees()
            );
            (*beta_up, *beta_down, *a_up, *a_down) = solve_fn(&case, theta_up, theta_down)
        };

        Zip::from(theta.slice(slice_up))
            .and(theta.slice(slice_down))
            .and(beta_up)
            .and(beta_down)
            .and(a_up)
            .and(a_down)
            .par_for_each(func);

        a_0.slice_mut(slice_down).assign(&a.slice(slice_up));

        VAWTSolution {
            case,
            n_streamtubes: self.n_streamtubes,
            theta,
            beta,
            a,
            a_0,
            epsilon: self.epsilon,
        }
    }

    fn get_case(&self) -> VAWTCase<'a> {
        VAWTCase {
            re: self.re,
            tsr: self.tsr,
            solidity: self.solidity,
            aerofoil: self.aerofoil,
        }
    }
}

/// Turbine settings for the VAWT case
#[derive(Debug, Clone)]
pub struct VAWTCase<'a> {
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
    case: VAWTCase<'a>,
    n_streamtubes: usize,
    theta: Array1<f64>,
    beta: Array1<f64>,
    a: Array1<f64>,
    a_0: Array1<f64>,
    epsilon: f64,
}

impl<'a> VAWTSolution<'a> {
    /// Torque ceofficient of the turbine
    pub fn c_torque(&self) -> f64 {
        let ct = Zip::from(&self.theta)
            .and(&self.beta)
            .and(&self.a)
            .and(&self.a_0)
            .fold(0.0, |acc, &theta, &beta, &a, &a_0| {
                let tube = StreamTube::new(theta, beta, a_0);
                let solution = StreamTubeSolution::new(self.case.clone(), tube, a);
                let ct = solution.c_tan();
                let w = solution.w();
                acc + ct * w.powi(2)
            });
        ct * self.case.solidity / (self.n_streamtubes as f64)
    }

    /// Power coefficient of the turbine
    pub fn c_power(&self) -> f64 {
        let ct = self.c_torque();
        ct * self.case.tsr
    }

    /// the pitch angle `beta` at the location `theta`
    pub fn beta(&self, theta: f64) -> f64 {
        Interp1D::new_unchecked(
            self.theta.view(),
            self.beta.view(),
            Linear::new().extrapolate(true),
        )
        .interp_scalar(theta)
        .unwrap()
    }

    /// the induction factor `a` at the location `theta`
    pub fn a(&self, theta: f64) -> f64 {
        Interp1D::new_unchecked(
            self.theta.view(),
            self.a.view(),
            Linear::new().extrapolate(true),
        )
        .interp_scalar(theta)
        .unwrap()
    }

    /// the upstream induction factor `a_0` at the location `theta`
    pub fn a_0(&self, theta: f64) -> f64 {
        Interp1D::new_unchecked(
            self.theta.view(),
            self.a_0.view(),
            Linear::new().extrapolate(true),
        )
        .interp_scalar(theta)
        .unwrap()
    }

    /// the difference between the wind thrust and the foil force (solution error)
    /// at the location `theta`
    pub fn thrust_error(&self, theta: f64) -> f64 {
        self.streamtube(theta).thrust_error()
    }

    /// tangential foil coefficient at the location `theta`
    ///
    /// coefficient of lift and drag evaluated in tangential direction
    pub fn c_tan(&self, theta: f64) -> f64 {
        self.streamtube(theta).c_tan()
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// the relative windspeed at the foil at location `theta`
    pub fn w(&self, theta: f64) -> f64 {
        self.streamtube(theta).w()
    }

    /// the angle of attac at the foil at location `theta`
    pub fn alpha(&self, theta: f64) -> f64 {
        self.streamtube(theta).alpha()
    }

    /// the local reynolds number at the foil at location `theta`
    pub fn re(&self, theta: f64) -> f64 {
        self.streamtube(theta).re()
    }

    fn streamtube(&self, theta: f64) -> StreamTubeSolution<'a> {
        let a_0 = self.a_0(theta);
        let a = self.a(theta);
        let beta = self.beta(theta);
        let tube = StreamTube::new(theta, beta, a_0);
        StreamTubeSolution::new(self.case.clone(), tube, a)
    }
}
