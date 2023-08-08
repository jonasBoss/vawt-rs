use std::{fs::File, collections::VecDeque};

use csv::{ReaderBuilder, StringRecord};
use ndarray::{Array2, Axis, Array3, s, Slice, Array1, array, concatenate, stack};
use ndarray_csv::Array2Reader;

mod areofoil;

fn main() {
    let mut files = VecDeque::from(["foo/cldAR/NACA0018Re0040.data", "foo/cldAR/NACA0018Re0080.data", "foo/cldAR/NACA0018Re0160.data"]);
    let re = array![40_000.0, 80_000.0, 160_000.0];

    let arrays: Vec<_> = files.into_iter().map(read_array).collect();
    let alpha = arrays[0].index_axis(Axis(1), 0).into_owned();
    let arrays_slice: Vec<_> = arrays.iter().map(|a|a.view()).collect();
    let mut data = stack(Axis(1), arrays_slice.as_slice()).unwrap();
    data.slice_axis_inplace(Axis(2), Slice::from(1..));

    println!("{data:?}");
    println!("{alpha:?}");
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