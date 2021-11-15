
extern crate image;
use ad_census::core::*;
use ad_census::utils;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use log::{debug, error, info, trace};

pub fn adcensus_matching_bench(c: &mut Criterion) {
    c.bench_function("ad-census match bench", |b| b.iter(|| {
        // 初始化日志
        //utils::init_log("trace");
        //info!("日志初始化完成!");
        let limg = image::open("images/left.png").unwrap().to_rgb8();;
        let rimg = image::open("images/right.png").unwrap().to_rgb8();;

        let option = ADCensusOption::new(0, 55)
        .set_lrcheck_thres(1.0f32)
        .set_do_lr_check(true)
        .set_do_filling(true)
        .set_irv_th(0.4)
        .set_irv_ts(20.0)
        .set_do_discontinuity_adjustment(false)
        .build();

        let mut stereo = ad_census::ADCensus::new(
            limg.width(),
            limg.height(),
            Some(option)
        ).unwrap();
        stereo.matching(&limg, &rimg);
    }));
}
criterion_group!(benches, adcensus_matching_bench);
criterion_main!(benches);