use crate::areofoil::Aerofoil;

#[derive(Debug)]
pub struct Turbine {
    /// Raynoldsnumber of the turbine
    pub re: f64,
    /// Tipspeed ratio of the turbine
    pub tsr: f64,
    /// Turbine solidity
    pub solidity: f64,
    /// Aerofoil
    pub foil: Aerofoil,
}
