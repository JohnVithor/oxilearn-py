use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::cell::Cell;

#[pyclass(name = "Counter")]
pub struct PyCounter {
    count: Cell<u64>,
    wraps: Py<PyAny>,
}

#[pymethods]
impl PyCounter {
    #[new]
    fn __new__(wraps: Py<PyAny>) -> Self {
        PyCounter {
            count: Cell::new(0),
            wraps,
        }
    }

    #[getter]
    fn count(&self) -> u64 {
        self.count.get()
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &PyTuple,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let old_count = self.count.get();
        let new_count = old_count + 1;
        self.count.set(new_count);
        let name = self.wraps.getattr(py, "__name__")?;

        println!("{} has been called {} time(s).", name, new_count);

        let ret = self.wraps.call(py, args, kwargs)?;

        Ok(ret)
    }
}
