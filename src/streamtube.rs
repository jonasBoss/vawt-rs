use std::{
    f64::consts::PI,
    ops::{Add, Sub},
};

use ndarray::{array, Array1};

use crate::{rot_mat, turbine::Turbine};

#[derive(Debug)]
struct StreamTube {
    /// Induction factor of the upwind streamtube.
    /// For upwind steamtubes (θ < π) this should be 0
    a_0: f64,
    /// Steamtube position in radians
    theta: f64,
    /// Pitch angle of the foil relative to the turbine tangent
    beta: f64,
}

impl StreamTube {
    pub fn new(theta: f64, beta: f64, a_0: f64) -> Self {
        Self { a_0, theta, beta }
    }

    /// solve the streamtube for induction factor a
    pub fn solve_a(&self, turbine: &Turbine, epsilon: f64) -> f64 {
        let mut a_range = (-2.0, 2.0);
        let mut err_range = (0.0, 0.0);
        err_range.0 = self.thrust_error(a_range.0, turbine);
        err_range.1 = self.thrust_error(a_range.1, turbine);
        if err_range.0 * err_range.1 > 0.0 {
            println!("strickland iteration");
            return self.a_strickland(turbine);
        }
        while (a_range.1 - a_range.0) > epsilon {
            let a = (a_range.1 - a_range.0) / 0.0;
            let err = self.thrust_error(a, turbine);

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
    pub fn thrust_error(&self, a: f64, turbine: &Turbine) -> f64 {
        let foil_force = self.foil_thrust(a, turbine);
        let wind_force = StreamTube::wind_thrust(a);

        foil_force - wind_force
    }

    /// the relative velocity magnitude `w`, the angle of attack `alpha` in radians
    /// and the local reynolds number `re` at the foil for a given induction factor a
    ///
    /// # returns
    /// `(w: f64, alpha: f64, re: f64)`
    pub fn w_alpha_re(&self, a: f64, turbine: &Turbine) -> (f64, f64, f64) {
        let w = self.w(a, turbine);
        let (w_x_floil, w_y_foil) = w.to_foil(self.theta, self.beta);

        let alpha = w_y_foil.atan2(w_x_floil) + PI / 2.0;
        let w = w.magnitude();
        let re = turbine.re * w;
        (w, alpha, re)
    }

    /// tangential foil coefficient and tangential force coeffitient
    ///
    /// # returns
    /// (c_tan: f64, cf_tan: f64)
    pub fn c_tan_cf_tan(&self, a: f64, turbine: &Turbine) -> (f64, f64) {
        let (w, alpha, re) = self.w_alpha_re(a, turbine);
        let (_, ct) = turbine
            .foil
            .cl_cd(alpha, re)
            .to_tangential(alpha, self.beta);
        (ct, ct * self.force_normalization(w, turbine))
    }

    fn a_strickland(&self, turbine: &Turbine) -> f64 {
        let mut a = 0.0;
        for _ in 0..10 {
            let c_s = self.foil_thrust(a, turbine);
            let a_new = 0.25 * c_s + a.powi(2);
            if a_new < 1.0 {
                a = a_new;
            } else {
                a = 1.0;
            }
        }
        a
    }

    fn foil_thrust(&self, a: f64, turbine: &Turbine) -> f64 {
        let (w, alpha, re) = self.w_alpha_re(a, turbine);

        let cl_cd = turbine.foil.cl_cd(alpha, re);
        let (_, force_coeff) = cl_cd.to_global(alpha, self.beta, self.theta);

        -force_coeff * self.force_normalization(w, turbine)
    }

    fn force_normalization(&self, w: f64, turbine: &Turbine) -> f64 {
        (w / self.c_0().magnitude()).powi(2) * turbine.solidity / (PI * self.theta.sin().abs())
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
    fn c_0(&self) -> Velocity {
        Velocity::from_global(0.0, -(1.0 - 2.0 * self.a_0))
    }

    /// windspeed at the foil in negative y direction
    fn c_1(&self, a: f64) -> Velocity {
        let magnitude = (1.0 - 2.0 * self.a_0) * (1.0 - a);
        Velocity::from_global(0.0, -magnitude)
    }

    /// relative velocity at foil in global xy coordinates
    fn w(&self, a: f64, turbine: &Turbine) -> Velocity {
        let u = turbine.tsr;
        self.c_1(a) - Velocity::from_tangential(0.0, u, self.theta)
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

    pub fn to_tangential(&self, theta: f64) -> (f64, f64) {
        let target = rot_mat(theta).dot(&self.0);
        (target[0], target[1])
    }

    pub fn to_foil(&self, theta: f64, beta: f64) -> (f64, f64) {
        let target = rot_mat(theta + beta).dot(&self.0);
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
