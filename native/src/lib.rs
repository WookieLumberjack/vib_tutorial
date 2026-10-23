//! Rust kernels for vib_tutorial's hot loops, imported as `vib_tutorial._native`.
//!
//! Each kernel mirrors Python code that stays the reference implementation
//! (`vib_tutorial.core.native` picks between them), and the tests compare the two.

mod film;
mod linalg;

use pyo3::prelude::*;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ECCENTRICITY_MAX", film::ECCENTRICITY_MAX)?;
    m.add_function(wrap_pyfunction!(film::pi_film, m)?)?;
    m.add_function(wrap_pyfunction!(film::full_film_damping, m)?)?;
    m.add_class::<film::FilmSteps>()?;
    m.add_function(wrap_pyfunction!(linalg::expm_py, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::foh_discretize_py, m)?)?;
    Ok(())
}
