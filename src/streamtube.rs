use std::{
    f64::consts::PI,
    ops::{Add, Sub},
};

use argmin::core::CostFunction;
use ndarray::{array, Array1};

use crate::{rot_mat, VAWTCase};

#[derive(Debug)]
pub struct StreamTube {
    /// Induction factor of the upwind streamtube.
    /// For upwind steamtubes (θ < π) this should be 0
    a_0: f64,
    /// Steamtube position in radians
    theta: f64,
    /// Pitch angle of the foil relative to the turbine tangent
    beta: f64,
}

impl StreamTube {
    /// create a new streamtube at the turbine position `theta` (in radians)
    /// with the pitch angle `beta` (in radians) and an upstream induction
    /// factor `a_0`. For upstream streamtubes this is likely `0.0`
    pub fn new(theta: f64, beta: f64, a_0: f64) -> Self {
        Self { a_0, theta, beta }
    }

    /// solve the streamtube for induction factor a
    pub fn solve_a(&self, case: &VAWTCase, epsilon: f64) -> f64 {
        let mut a_range = (-2.0, 2.0);
        let mut err_range = (0.0, 0.0);
        err_range.0 = self.thrust_error(a_range.0, case);
        err_range.1 = self.thrust_error(a_range.1, case);
        if err_range.0 * err_range.1 > 0.0 {
            return self.a_strickland(case);
        }
        while (a_range.1 - a_range.0) > epsilon {
            let a = a_range.0 + (a_range.1 - a_range.0) / 2.0;
            let err = self.thrust_error(a, case);

            if err_range.0 * err <= 0.0 {
                a_range.1 = a;
                err_range.1 = err;
            } else {
                a_range.0 = a;
                err_range.0 = err;
            }
        }

        a_range.0 + (a_range.1 - a_range.0) / 2.0
    }

    /// the difference between the wind thrust and the foil force
    /// for a given induction factor a
    ///
    /// for good solutions this should be small
    fn thrust_error(&self, a: f64, case: &VAWTCase) -> f64 {
        let foil_force = self.foil_thrust(a, case);
        let wind_force = StreamTube::wind_thrust(a);

        foil_force - wind_force
    }

    /// the relative velocity magnitude `w`, the angle of attack `alpha` in radians
    /// and the local reynolds number `re` at the foil for a given induction factor a
    ///
    /// # returns
    /// `(w: f64, alpha: f64, re: f64)`
    fn w_alpha_re(&self, a: f64, case: &VAWTCase) -> (f64, f64, f64) {
        let w = self.w_vec(a, case);
        let (w_x_floil, w_y_foil) = w.to_foil(self.theta, self.beta);

        let alpha = w_y_foil.atan2(w_x_floil) + PI / 2.0;
        let w = w.magnitude();
        let re = case.re * w;
        (w, alpha, re)
    }

    /// tangential foil coefficient
    pub(crate) fn c_tan(&self, a: f64, case: &VAWTCase) -> f64 {
        let (_w, alpha, re) = self.w_alpha_re(a, case);
        case
            .aerofoil
            .cl_cd(alpha, re)
            .to_tangential(alpha, self.beta)
            .1
    }

    fn a_strickland(&self, case: &VAWTCase) -> f64 {
        let mut a = 0.0;
        for _ in 0..10 {
            let c_s = self.foil_thrust(a, case);
            let a_new = 0.25 * c_s + a.powi(2);
            if a_new < 1.0 {
                a = a_new;
            } else {
                a = 1.0;
            }
        }
        a
    }

    fn foil_thrust(&self, a: f64, case: &VAWTCase) -> f64 {
        let (w, alpha, re) = self.w_alpha_re(a, case);

        let cl_cd = case.aerofoil.cl_cd(alpha, re);
        let (_, force_coeff) = cl_cd.to_global(alpha, self.beta, self.theta);

        -force_coeff * (w / self.c_0()).powi(2) * case.solidity / (PI * self.theta.sin().abs())
    }

    /// Thrust coefficient by momentum theory or Glauert empirical formula
    ///
    /// A crude straight line approximation for Glauert formula is used
    /// between 0.4 < a < 1.0,  0.96 < CtubeThru < 2.0
    fn wind_thrust(a: f64) -> f64 {
        if a < 0.4 {
            4.0 * a * (1.0 - a)
        } else {
            26.0 / 15.0 * a + 4.0 / 15.0
        }
    }

    /// reference windspeed
    fn c_0(&self) -> f64 {
        1.0 - 2.0 * self.a_0
    }

    /// windspeed at the foil in negative y direction
    fn c_1_vec(&self, a: f64) -> Velocity {
        let magnitude = self.c_0() * (1.0 - a);
        Velocity::from_global(0.0, -magnitude)
    }

    /// relative velocity at foil in global xy coordinates
    fn w_vec(&self, a: f64, case: &VAWTCase) -> Velocity {
        let u = case.tsr;
        self.c_1_vec(a) - Velocity::from_tangential(0.0, u, self.theta)
    }
}

pub(crate) struct OptimizeBeta<'a> {
    pub(crate) case: &'a VAWTCase<'a>,
    pub(crate) epsilon: f64,
    pub(crate) theta_up: f64,
    pub(crate) theta_down: f64,
}
impl<'a> OptimizeBeta<'a> {
    pub(crate) fn new(
        case: &'a VAWTCase<'a>,
        epsilon: f64,
        theta_up: f64,
        theta_down: f64,
    ) -> Self {
        Self {
            case,
            epsilon,
            theta_up,
            theta_down,
        }
    }

    pub(crate) fn cost(&self, param: &Array1<f64>) -> f64 {
        let tube_up = StreamTube::new(self.theta_up, param[0], 0.0);
        let a_up = tube_up.solve_a(self.case, self.epsilon);
        let tube_down = StreamTube::new(self.theta_down, param[1], a_up);
        let a_down = tube_down.solve_a(self.case, self.epsilon);

        let w_up = tube_up.w_vec(a_up, self.case).magnitude();
        let w_down = tube_down.w_vec(a_down, self.case).magnitude();

        let ct = tube_up.c_tan(a_up, self.case) * w_up.powi(2)
            + tube_down.c_tan(a_down, self.case) * w_down.powi(2);
        1.0 - ct
    }
}

impl CostFunction for OptimizeBeta<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(self.cost(param))
    }
}

pub struct StreamTubeSolution<'a> {
    case: VAWTCase<'a>,
    tube: StreamTube,
    a: f64,
}

impl<'a> StreamTubeSolution<'a> {
    pub(crate) fn new(case: VAWTCase<'a>, tube: StreamTube, a: f64) -> Self {
        StreamTubeSolution { case, tube, a }
    }

    /// the induction factor of the solution
    pub fn a(&self) -> f64 {
        self.a
    }

    /// the induction factor of the upstream streamtube
    pub fn a_0(&self) -> f64 {
        self.tube.a_0
    }

    /// the pitch angel in radians
    pub fn beta(&self) -> f64 {
        self.tube.beta
    }

    /// the stramtube location in radinas
    pub fn theta(&self) -> f64 {
        self.tube.theta
    }

    /// the relative windspeed at the foil
    pub fn w(&self) -> f64 {
        self.tube.w_vec(self.a, &self.case).magnitude()
    }

    /// the angle of attac at the foil
    pub fn alpha(&self) -> f64 {
        let w = self.tube.w_vec(self.a, &self.case);
        let (w_x_floil, w_y_foil) = w.to_foil(self.tube.theta, self.tube.beta);

        w_y_foil.atan2(w_x_floil) + PI / 2.0
    }

    /// the local reynolds number at the foil
    pub fn re(&self) -> f64 {
        self.w() * self.case.re
    }

    /// the difference between the wind thrust and the foil force of the solution
    pub fn thrust_error(&self) -> f64 {
        self.tube.thrust_error(self.a, &self.case)
    }

    /// tangential foil coefficient
    ///
    /// coefficient of lift and drag evaluated in tangential direction
    pub fn c_tan(&self) -> f64 {
        self.tube.c_tan(self.a, &self.case)
    }
}

/// A velocity in global coordinates
#[derive(Debug)]
struct Velocity(Array1<f64>);

impl Velocity {
    pub fn from_global(x: f64, y: f64) -> Self {
        Self(array![x, y])
    }

    pub fn from_tangential(x: f64, y: f64, theta: f64) -> Self {
        Velocity(rot_mat(theta).dot(&array![x, y]))
    }

    pub fn to_foil(&self, theta: f64, beta: f64) -> (f64, f64) {
        let target = rot_mat(-theta - beta).dot(&self.0);
        (target[0], target[1])
    }

    pub fn magnitude(&self) -> f64 {
        (self.0[0].powi(2) + self.0[1].powi(2)).sqrt()
    }
}

impl Sub for Velocity {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let Velocity(lhs) = self;
        let Velocity(rhs) = rhs;
        Velocity(lhs - rhs)
    }
}

impl Add for Velocity {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let Velocity(lhs) = self;
        let Velocity(rhs) = rhs;
        Velocity(lhs + rhs)
    }
}
