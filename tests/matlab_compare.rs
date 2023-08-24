use std::{error::Error, f64::consts::PI, fs::File};

use approx::{assert_abs_diff_eq, assert_relative_eq};
use csv::ReaderBuilder;
use ndarray::{s, Array, Array1, Array2, ArrayView1, Axis, Zip};
use ndarray_csv::Array2Reader;
use vawt::{areofoil::Aerofoil, VAWTSolution, VAWTSolver, Verbosity};

fn load_naca_0018() -> Result<Aerofoil, Box<dyn Error>> {
    fn read_array(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .trim(csv::Trim::All)
            .delimiter(b',')
            .from_reader(file);

        let mut arr: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
        for mut datapoint in arr.axis_iter_mut(Axis(0)) {
            datapoint[0] = datapoint[0].to_radians();
        }
        Ok(arr)
    }
    Ok(Aerofoil::builder()
        .add_data_row(read_array("tests/NACA0018/NACA0018Re0080.data")?, 80_000.0)?
        .add_data_row(read_array("tests/NACA0018/NACA0018Re0040.data")?, 40_000.0)?
        .add_data_row(read_array("tests/NACA0018/NACA0018Re0160.data")?, 160_000.0)?
        .set_aspect_ratio(12.8)
        .update_aspect_ratio(true)
        .symmetric(true)
        .build()?)
}

#[derive(Debug)]
struct MatlabSolution(Array2<f64>);

impl MatlabSolution {
    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut arr: Array2<f64> = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .delimiter(b'\t')
            .from_reader(file)
            .deserialize_array2_dynamic()?;

        for mut datapoint in arr.axis_iter_mut(Axis(0)) {
            datapoint[0] = datapoint[0].to_radians();
            datapoint[3] = datapoint[3].to_radians();
            datapoint[4] = datapoint[4] * 1e5;
        }
        Ok(MatlabSolution(arr))
    }

    fn theta(&self) -> ArrayView1<f64> {
        self.0.slice(s![.., 0])
    }

    fn a(&self) -> ArrayView1<f64> {
        self.0.slice(s![.., 1])
    }

    fn n_streamtubes(&self) -> usize {
        self.0.shape()[0]
    }

    fn w(&self) -> ArrayView1<f64> {
        self.0.slice(s![.., 2])
    }

    fn alpha(&self) -> ArrayView1<f64> {
        self.0.slice(s![.., 3])
    }

    fn re(&self) -> ArrayView1<f64> {
        self.0.slice(s![.., 4])
    }
}

fn naca0018_solution(
) -> Result<(&'static Aerofoil, MatlabSolution, VAWTSolution<'static>), Box<dyn Error>> {
    let aerofoil = load_naca_0018()?;
    let aerofoil = Box::leak(Box::new(aerofoil));
    let matlab_solution = MatlabSolution::load("tests/matlab_NACA0018_tsr-3.25.txt")?;

    let solution = VAWTSolver::new(aerofoil)
        .re(31_300.0)
        .solidity(0.3525)
        .n_streamtubes(matlab_solution.n_streamtubes())
        .tsr(3.25)
        .verbosity(Verbosity::Silent)
        .solve_with_beta(0.0);
    Ok((aerofoil, matlab_solution, solution))
}

fn map_theta<'a, F: Fn(f64) -> f64>(matlab_solution: &MatlabSolution, f: F) -> Array1<f64> {
    Array::from_iter(matlab_solution.theta().iter().map(move |&theta| f(theta)))
}

#[test]
fn test_naca_0018_induction_factor() -> Result<(), Box<dyn Error>> {
    let (_, matlab_solution, solution) = naca0018_solution()?;
    let rust_a = map_theta(&matlab_solution, |t| solution.a(t));
    assert_relative_eq!(
        rust_a,
        matlab_solution.a(),
        epsilon = 2.0 * solution.epsilon(),
        max_relative = 0.01
    );
    Ok(())
}

#[test]
fn test_naca_0018_w() -> Result<(), Box<dyn Error>> {
    let (_, matlab_solution, solution) = naca0018_solution()?;
    let rust_w = map_theta(&matlab_solution, |theta| solution.w_alpha_re(theta).0);
    assert_relative_eq!(
        rust_w,
        matlab_solution.w(),
        epsilon = 0.01,
        max_relative = 0.01
    );
    Ok(())
}

#[test]
fn test_naca_0018_alpha() -> Result<(), Box<dyn Error>> {
    let (_, matlab_solution, solution) = naca0018_solution()?;
    let rust_alpha = map_theta(&matlab_solution, |theta| solution.w_alpha_re(theta).1);
    assert_relative_eq!(
        rust_alpha,
        matlab_solution.alpha(),
        epsilon = 0.01,
        max_relative = 0.01
    );
    Ok(())
}

#[test]
fn test_naca_0018_re() -> Result<(), Box<dyn Error>> {
    let (_, matlab_solution, solution) = naca0018_solution()?;
    let rust_re = map_theta(&matlab_solution, |theta| solution.w_alpha_re(theta).2);
    assert_relative_eq!(
        rust_re,
        matlab_solution.re(),
        epsilon = 0.01,
        max_relative = 0.01
    );
    Ok(())
}
