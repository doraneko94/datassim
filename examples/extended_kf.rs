use eom::*;
use eom::traits::*;
use ndarray::*;
use ndarray_linalg::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::fs::File;
use std::io::{BufRead, Write, BufReader, BufWriter};

const TIME_STEP: usize = 14_600;
const F: f64 = 8.0;
const N: usize = 40;
const M: usize = 40;
const DELTA: f64 = 1e-5;

#[allow(bare_trait_objects)]
fn main() -> Result<(), Box<std::error::Error>> {
    let mut h: Array2<f64> = Array2::zeros((M, N));
    for i in 0..M { h[[i, i]] = 1.0 }
    let r: Array2<f64> = Array2::eye(N);

    let mut u: Array1<f64> = Array::from((0..N).map(|_| thread_rng().sample(StandardNormal))
                                                    .collect::<Vec<f64>>());
    u += F;

    let dt = 0.05;
    let eom = ode::Lorenz96 { f: F, n: N };
    let mut teo = explicit::RK4::new(eom, dt);
    let ts = adaptor::time_series(u, &mut teo);

    let mut ua: Array1<f64> = Array1::zeros(N);
    for (t, v) in ts.take(TIME_STEP).enumerate() {
        if t == TIME_STEP - 1 { ua = v.to_owned(); }
    }
    let mut pa = 25.0 * Array2::eye(N);

    let mut truestate = BufReader::new(File::open("L96_truestate.txt")?);
    let mut observation = BufReader::new(File::open("L96_observation.txt")?);
    let mut output = BufWriter::new(File::create("L96_EKF_output_noinflation.txt")?);

    for i in 0..TIME_STEP {
        let mut ua_c = ua.to_owned();
        let uf = teo.iterate(&mut ua_c).to_owned();

        let mut s_true = String::new();
        let mut s_obs = String::new();
        truestate.read_line(&mut s_true)?;
        observation.read_line(&mut s_obs)?;
        let u_true = arr1(&s_true.trim()
                                 .split(",")
                                 .map(|e| e.parse().ok().unwrap())
                                 .collect::<Vec<f64>>()[1..]);
        let u_obs = arr1(&s_obs.trim()
                               .split(",")
                               .map(|e| e.parse().ok().unwrap())
                               .collect::<Vec<f64>>()[1..]);

        let mut jm = Array2::zeros((N, N));
        for j in 0..M {
            let mut ua_h = ua.to_owned();
            let mut ua_0 = ua.to_owned();
            ua_h[j] += DELTA;
            let f_h = teo.iterate(&mut ua_h).to_owned();
            let f = teo.iterate(&mut ua_0).to_owned();
            let df = (f_h - f) / DELTA;
            
            for ii in 0..N {
                jm[[ii, j]] = df[ii];
            }
        }
        let mut pf: Array2<f64> = jm.dot(&pa);
        pf = pf.dot(&jm.t()) * 1.1;
        let pf_ht = pf.dot(&h.t());
        let k = pf_ht.dot(&(h.dot(&pf_ht) + &r).inv()?);
        ua = uf.to_owned() + k.dot(&(u_obs.to_owned() - h.dot(&uf)));
        pa = (Array2::eye(N) - k.dot(&h)).dot(&pf);

        let out = format!("{:.5},{:.5},{:.5},{:.5}\n",
                          i as f64 / 40.0,
                          ((0..N).map(|i| pa[[i, i]]).sum::<f64>() / N as f64).sqrt(),
                          (u_true.to_owned() - uf).norm_l2() / (N as f64).sqrt(),
                          (u_obs - u_true).norm_l2() / (N as f64).sqrt());
        output.write_all(out.as_bytes())?;
    }

    Ok(())
}