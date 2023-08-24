use std::{error::Error, f64::consts::PI, fs::File};

use csv::ReaderBuilder;
use ndarray::{Array, Array2, Axis};
use ndarray_csv::Array2Reader;

use vawt::{areofoil::Aerofoil, VAWTSolver, Verbosity};

fn main() -> Result<(), Box<dyn Error>> {
    let aerofoil = Aerofoil::builder()
        .add_data_row(
            read_array("examples/NACA0018/NACA0018Re0080.data")?,
            80_000.0,
        )?
        .add_data_row(
            read_array("examples/NACA0018/NACA0018Re0040.data")?,
            40_000.0,
        )?
        .add_data_row(
            read_array("examples/NACA0018/NACA0018Re0160.data")?,
            160_000.0,
        )?
        .set_aspect_ratio(12.8)
        .update_aspect_ratio(true)
        .symmetric(true)
        .build()?;

    let n = 72;
    let d_t_half = PI / n as f64;
    let _theta = Array::linspace(d_t_half, 2.0 * PI - d_t_half, n);

    let mut solver = VAWTSolver::new(&aerofoil);
    solver
        .re(31_300.0)
        .solidity(0.3525)
        .n_streamtubes(n)
        .verbosity(Verbosity::Silent);
    let _solution = solver.tsr(3.25).solve_with_beta(0.0);
    Ok(())
}

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
