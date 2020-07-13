use eom::*;
use eom::traits::*;
use ndarray::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

const F: f64 = 8.0;
const N: usize = 40;

fn main() {
    let u: Array1<f64> = Array::from((0..N).map(|_| thread_rng().sample(StandardNormal))
                                               .collect::<Vec<f64>>());
    let dt = 0.01;
    let eom = ode::Lorenz96 { f: F, n: N };
    let mut teo = explicit::RK4::new(eom, dt);
    let ts = adaptor::time_series(u, &mut teo);
    let end_time = 2_000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}