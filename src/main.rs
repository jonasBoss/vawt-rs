use std::{error::Error, fs::File, time::Instant};

use csv::ReaderBuilder;

use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;

use vawt::{areofoil::Aerofoil, VAWTSolver};

fn main() -> Result<(), Box<dyn Error>> {
    let aerofoil = load_naca_0018()?;
    let testcase = setup_solver(&aerofoil);

    let start = Instant::now();
    for _ in 0..10_000 {
        testcase.solve_with_beta(0.0);
    }
    let duration = start.elapsed();
    println!("Duration for 10'000 solutions: {} microseconds", duration.as_micros());
    Ok(())
}

fn load_naca_0018() -> Result<Aerofoil, Box<dyn Error>> {
    Ok(Aerofoil::builder()
        .add_data_row(read_array("tests/NACA0018/NACA0018Re0080.data")?, 80_000.0)?
        .add_data_row(read_array("tests/NACA0018/NACA0018Re0040.data")?, 40_000.0)?
        .add_data_row(read_array("tests/NACA0018/NACA0018Re0160.data")?, 160_000.0)?
        .set_aspect_ratio(12.8)
        .update_aspect_ratio(true)
        .symmetric(true)
        .build()?)
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

fn setup_solver(aerofoil: &Aerofoil) -> VAWTSolver{
    let mut testcase = VAWTSolver::new(&aerofoil);
    testcase
        .re(31_300.0)
        .solidity(0.3525)
        .n_streamtubes(72)
        .tsr(3.25);
    testcase
}