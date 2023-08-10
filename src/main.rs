use std::{collections::VecDeque, fs::File};

use csv::{ReaderBuilder, StringRecord};
use ndarray::{array, concatenate, s, stack, Array1, Array2, Array3, Axis, Slice};
use ndarray_csv::Array2Reader;

use crate::areofoil::Aerofoil;

mod areofoil;

fn main() {
    let files = Vec::from([
        "foo/cldAR/NACA0018Re0080.data",
        "foo/cldAR/NACA0018Re0040.data",
        "foo/cldAR/NACA0018Re0160.data",
    ]);
    let re = array![80_000.0, 40_000.0, 160_000.0];

    let mut builder = Aerofoil::builder();
    for (file, re) in files.into_iter().zip(re) {
        builder = builder.add_data_row(read_array(file), re)
    }

    let aerofoil = builder
        .set_aspect_ratio(12.8)
        .update_aspect_ratio(true)
        .build()
        .unwrap();
    println!("{aerofoil:?}")
}

fn read_array(path: &str) -> Array2<f64> {
    let file = File::open(path).unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .delimiter(b',')
        .from_reader(file);

    reader.deserialize_array2_dynamic().unwrap()
}
