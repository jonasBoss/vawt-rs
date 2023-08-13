#[derive(Debug)]
pub struct TurbineOperatingConditions {
    /// Raynoldsnumber of the turbine
    pub re: f64,
    /// Tipspeed ratio of the turbine
    pub tsr: f64,
}
