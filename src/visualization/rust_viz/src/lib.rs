use pyo3::prelude::*;
use plotters::prelude::*;
use std::collections::VecDeque;

#[pyclass]
struct VizEngine {
    train_losses: VecDeque<f32>,
    val_losses: VecDeque<f32>,
    accuracies: VecDeque<f32>,
    learning_rates: VecDeque<f32>,
}

#[pymethods]
impl VizEngine {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(VizEngine {
            train_losses: VecDeque::with_capacity(100),
            val_losses: VecDeque::with_capacity(100),
            accuracies: VecDeque::with_capacity(100),
            learning_rates: VecDeque::with_capacity(100),
        })
    }

    fn update(&mut self, train_loss: f32, val_loss: f32, accuracy: f32, learning_rate: f32) -> PyResult<()> {
        // Update data
        if self.train_losses.len() >= 100 {
            self.train_losses.pop_front();
            self.val_losses.pop_front();
            self.accuracies.pop_front();
            self.learning_rates.pop_front();
        }
        
        self.train_losses.push_back(train_loss);
        self.val_losses.push_back(val_loss);
        self.accuracies.push_back(accuracy);
        self.learning_rates.push_back(learning_rate);

        // Create the plot
        let root = BitMapBackend::new("training_progress.png", (800, 600))
            .into_drawing_area();
        
        if let Err(e) = root.fill(&WHITE) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Drawing error: {}", e)
            ));
        }

        let mut chart = match ChartBuilder::on(&root)  // Added mut here
            .caption("Training Progress", ("sans-serif", 30))
            .margin(10)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(
                0f32..self.train_losses.len() as f32,
                0f32..100f32,
            ) {
                Ok(chart) => chart,
                Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Chart error: {}", e)
                )),
            };

        if let Err(e) = chart.configure_mesh().draw() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Mesh error: {}", e)
            ));
        }

        // Plot training loss
        if let Err(e) = chart.draw_series(LineSeries::new(
            self.train_losses.iter().enumerate().map(|(i, &v)| (i as f32, v * 100.0)),
            &RED,
        )).map(|line| line.label("Training Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED))) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Series error: {}", e)
            ));
        }

        // Plot validation loss
        if let Err(e) = chart.draw_series(LineSeries::new(
            self.val_losses.iter().enumerate().map(|(i, &v)| (i as f32, v * 100.0)),
            &BLUE,
        )).map(|line| line.label("Validation Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE))) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Series error: {}", e)
            ));
        }

        // Plot accuracy
        if let Err(e) = chart.draw_series(LineSeries::new(
            self.accuracies.iter().enumerate().map(|(i, &v)| (i as f32, v)),
            &GREEN,
        )).map(|line| line.label("Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN))) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Series error: {}", e)
            ));
        }

        if let Err(e) = chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Legend error: {}", e)
            ));
        }

        if let Err(e) = root.present() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Present error: {}", e)
            ));
        }

        Ok(())
    }
}

#[pymodule]
fn rust_viz(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VizEngine>()?;
    Ok(())
}