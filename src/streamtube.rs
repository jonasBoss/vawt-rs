use ndarray::Array1;

use crate::{
    areofoil::{Aerofoil, ClCd},
    rot_mat,
    turbine::TurbineOperatingConditions,
};

#[derive(Debug)]
struct StreamTube {
    /// Induction factor of the upwind streamtube.
    /// For upwind steamtubes (θ < 180°) this should be 0
    a_0: f64,
    /// Steamtube position
    θ: f64,
    /// Pitch angle of the foil relative to the turbine tangent
    β: f64,
}

impl StreamTube {
    pub fn new(θ: f64, β: f64, a_0: f64) -> Self {
        Self { a_0, θ, β }
    }

    /// calculate streamtube solution for the given tipspeed ratio
    pub fn solve(
        &self,
        _foil: &Aerofoil,
        operating_point: &TurbineOperatingConditions,
    ) -> StreamTubeSolution {
        let a = self.calculate_a(operating_point.tsr, 0.01);

        // reference windspeed
        let c_0 = 1.0 - 2.0 * self.a_0;
        // windspeed at wing
        let _c_1 = c_0 * (1.0 - a);
        // tangential wing velocity
        let _u = operating_point.tsr;

        todo!()
    }

    /// calculate the induction factor for the stramtube
    pub fn calculate_a(&self, _tsr: f64, ε: f64) -> f64 {
        let a_range = (-2.0, 2.0);
        while (a_range.1 - a_range.0) > ε {
            todo!()
        }

        a_range.0 + (a_range.1 - a_range.0) / 2.0
    }

    /// the difference between the wind thrust and the foil force
    /// this needs to be minimized
    pub fn thrust_error(&self, _a: f64) -> f64 {
        todo!()
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

    /// Force coefficient acting on the aerofoil in normal and tangent direction
    pub fn cn_ct(&self, alpha: f64, re_1: f64, foil: &Aerofoil) -> Array1<f64> {
        let _cld = foil.cl_cd(alpha, re_1);
        todo!()
    }
}

#[derive(Debug)]
struct StreamTubeSolution {
    /// Induction factor of the streamtube
    a: f64,
}

#[derive(Debug)]
struct CnCt(Array1<f64>);

impl CnCt {
    pub fn from_clcd(cl_cd: &ClCd, alpha: f64, beta: f64) -> Self {
        let ClCd(cl_cd) = cl_cd;
        let cn_ct = rot_mat(alpha + beta).dot(cl_cd);
        Self(cn_ct)
    }
}
