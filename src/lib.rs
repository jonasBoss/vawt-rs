use ndarray::{array, Array2};

pub mod areofoil;
pub mod streamtube;
pub mod turbine;

/// 2d rotation matrix for angle phi (in radians)
fn rot_mat(phi: f64) -> Array2<f64> {
    array![[phi.cos(), (-phi).sin()], [phi.sin(), phi.cos()],]
}
