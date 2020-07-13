use eom::*;
use eom::traits::*;
use gnuplot::{Figure, Caption, Color};
use ndarray::*;
use rand::prelude::*;
use rand_distr::Normal;

const F: f64 = 8.0;
const N: usize = 40;

const EPSILON: f64 = 1e-3;
const SAMPLE: usize = 40;

fn main() {
    let nd = Normal::<f64>::new(0.0, 1.0).unwrap();
    let u: Array1<f64> = Array::from((0..N).map(|_| thread_rng().sample(nd))
                                               .collect::<Vec<f64>>());
    let dt = 0.01;
    let eom = ode::Lorenz96 { f: F, n: N };
    let mut teo = explicit::RK4::new(eom, dt);
    let ts = adaptor::time_series(u, &mut teo);
    let end_time = 10_000;

    let mut sol: Array2<f64> = Array2::zeros((end_time, N));
    for (i, v) in ts.take(end_time).enumerate() {
        for j in 0..N {
            sol[[i, j]] = v[j];
        }
    }

    let end_time = 100;
    let mut error: Array2<f64> = Array::zeros((end_time, SAMPLE));

    for i in 0..SAMPLE {
        let u_attract = (0..N).map(|j| sol[[3000+100*i, j]])
                              .collect::<Vec<f64>>();
        let u_attract_perturb = u_attract.iter()
                                         .map(|&e| e + EPSILON * thread_rng().sample(nd))
                                         .collect::<Vec<f64>>();
        
        let mut teo_attract = explicit::RK4::new(eom, dt);
        let ts_attract = adaptor::time_series(arr1(&u_attract), &mut teo_attract);

        let mut teo_attract_perturb = explicit::RK4::new(eom, dt);
        let ts_attract_perturb = adaptor::time_series(arr1(&u_attract_perturb), &mut teo_attract_perturb);

        for (t, (v0, v1)) in ts_attract.take(end_time)
                                .zip(ts_attract_perturb.take(end_time))
                                .enumerate() {
            for j in 0..N {
                let e = v0[j] - v1[j];
                error[[t, i]] += e * e;
            }
            error[[t, i]] /= N as f64;
        }
    }

    let x = (1..101).map(|i| i as f64).collect::<Vec<f64>>();
    let y = (0..end_time).map(|i| {
                                (0..SAMPLE).fold(0.0, |m, j| m + error[[i, j]]).sqrt()
                            })
                         .collect::<Vec<f64>>();
    let mut fg = Figure::new();
    fg.axes2d()
      .lines(&x, &y, &[Caption("y1"), Color("blue")]);
    fg.echo_to_file("scale.plt");
}