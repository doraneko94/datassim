use eom::*;
use eom::traits::*;
use ndarray::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::fs::File;
use std::io::{Write, BufWriter};

const TIME_STEP: usize = 14_600;
const F: f64 = 8.0;
const N: usize = 40;

#[allow(bare_trait_objects)]
fn main() -> Result<(), Box<std::error::Error>> {
    let u: Array1<f64> = Array::from((0..N).map(|_| thread_rng().sample(StandardNormal))
                                               .collect::<Vec<f64>>());
    let dt = 0.05;
    let eom = ode::Lorenz96 { f: F, n: N };
    let mut teo = explicit::RK4::new(eom, dt);
    let ts = adaptor::time_series(u, &mut teo);

    let mut truestate = BufWriter::new(File::create("L96_truestate.txt")?);
    let mut observation = BufWriter::new(File::create("L96_observation.txt")?);

    for (t, v) in ts.take(TIME_STEP * 2).enumerate() {
        if t >= TIME_STEP {
            let i = (t - TIME_STEP) as f64;
            truestate.write_all(format!("{}", i / 40.0).as_bytes())?;
            observation.write_all(format!("{}", i / 40.0).as_bytes())?;
            for &e in v.iter() {
                truestate.write_all(format!(",{:.5}", e).as_bytes())?;
                let noise: f64 = thread_rng().sample(StandardNormal);
                observation.write_all(format!(",{:.5}", e + noise).as_bytes())?;
            }
            truestate.write_all("\n".as_bytes())?;
            observation.write_all("\n".as_bytes())?;
        }
    }
    Ok(())
}