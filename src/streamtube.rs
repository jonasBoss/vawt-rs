use std::{ops::{SubAssign, Sub, Add}, f64::consts::PI};

use ndarray::{Array1, array};

use crate::{
    areofoil::{Aerofoil, ClCd},
    rot_mat,
    turbine::Turbine,
};

#[derive(Debug)]
struct StreamTube {
    /// Induction factor of the upwind streamtube.
    /// For upwind steamtubes (θ < 180°) this should be 0
    a_0: f64,
    /// Steamtube position
    theta: f64,
    /// Pitch angle of the foil relative to the turbine tangent
    beta: f64,
}

impl StreamTube {
    pub fn new(theta: f64, beta: f64, a_0: f64) -> Self {
        Self { a_0, theta, beta }
    }

    /// calculate streamtube solution for the given tipspeed ratio
    pub fn solve(
        &self,
        turbine: &Turbine,
    ) -> StreamTubeSolution {
        let a = self.calculate_a(turbine, 0.01);

        // reference windspeed
        let c_0 = 1.0 - 2.0 * self.a_0;
        // windspeed at wing
        let _c_1 = c_0 * (1.0 - a);
        // tangential wing velocity
        let _u = turbine.tsr;

        todo!()
    }

    /// calculate the induction factor for the stramtube
    pub fn calculate_a(&self, turbine: &Turbine, epsilon: f64) -> f64 {
        let mut a_range = (-2.0, 2.0);
        let mut err_range = (0.0,0.0);
        err_range.0 = self.thrust_error(a_range.0, turbine);
        err_range.1 = self.thrust_error(a_range.1, turbine);
        if err_range.0 * err_range.1 > 0.0 {
            println!("strickland iteration");
            return self.a_strickland(turbine);
        }
        while(a_range.1 - a_range.0) > epsilon {
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

    fn a_strickland(&self, turbine: &Turbine) -> f64 {
        let mut a = 0.0;
        for _ in 0..10 {
            let c_s = self.foil_thrust(a, turbine);
            let a_new = 0.25*c_s + a.powi(2);
            if a_new < 1.0 {
                a = a_new;
            }else {
                a = 1.0;
            }
        }
        a
    }

    /// the difference between the wind thrust and the foil force
    /// this needs to be minimized
    pub fn thrust_error(&self, a: f64, turbine: &Turbine) -> f64 {
        let foil_force = self.foil_thrust(a, turbine);
        let wind_force = StreamTube::wind_thrust(a);

        foil_force - wind_force
    }

    fn foil_thrust(&self, a: f64, turbine: &Turbine) -> f64 {
        let w = self.w(a, turbine);
        let (w_x_floil, w_y_foil) = w.to_foil(self.theta, self.beta);
        let w = w.magnitude();
        
        let alpha = w_y_foil.atan2(w_x_floil) + PI / 2.0;
        let re = turbine.re * w;

        let cl_cd = turbine.foil.cl_cd(alpha, re);
        let (_, force_coeff) = cl_cd.to_global(alpha, self.beta, self.theta);

        - force_coeff * (w / self.c_0().magnitude()).powi(2) * turbine.solidity / (PI * self.theta.sin().abs())
    }

    /// Thrust coefficient by momentum theory or Glauert empirical formula
    ///
    /// A crude straight line approximation for Glauert formula is used
    /// between 0.4 < a < 1.0,  0.96 < CtubeThru < 2.0
    pub fn wind_thrust(a: f64) -> f64 {
        if a < 0.4 {
            4.0 * a * (1.0 - a)
        } else {
            26.0 / 15.0 * a + 4.0 / 15.0
        }
    }

    /// reference windspeed 
    fn c_0(&self) -> Velocity {
        Velocity::from_global(0.0, -(1.0-2.0*self.a_0))
    }

    /// windspeed at the foil in negative y direction
    fn c_1(&self, a: f64) -> Velocity {
        let magnitude = (1.0-2.0*self.a_0) * (1.0-a);
        Velocity::from_global(0.0, - magnitude)
    }

    /// relative velocity at foil in global xy coordinates
    fn w(&self, a: f64, turbine: &Turbine) -> Velocity{
        let u = turbine.tsr;
        self.c_1(a) - Velocity::from_tangential(0.0, u, self.theta)
    }
}

#[derive(Debug)]
struct StreamTubeSolution {
    /// Induction factor of the streamtube
    a: f64,
}

/// Normal and Tangent coefficients
#[derive(Debug)]
struct CnCt(Array1<f64>);

impl CnCt {
    pub fn from_clcd(cl_cd: &ClCd, alpha: f64, beta: f64) -> Self {
        let ClCd(cl_cd) = cl_cd;
        let cn_ct = rot_mat(alpha + beta).dot(cl_cd);
        Self(cn_ct)
    }
}

/// A velocity in global coordinates
#[derive(Debug)]
struct Velocity(Array1<f64>);

impl Velocity {
    pub fn from_global(x: f64, y: f64) -> Self {
        Self(array![x,y])
    }

    pub fn from_tangential(x: f64, y: f64, theta: f64) -> Self {
        Velocity(rot_mat(theta).dot(&array![x,y]))
    }

    pub fn to_tangential(&self, theta: f64) -> (f64, f64){
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