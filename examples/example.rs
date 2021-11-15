extern crate image;
use ad_census::core::*;
use ad_census::utils;

use log::{debug, error, info, trace};
pub fn main() {
    // 初始化日志
    utils::init_log("trace");
    //info!("日志初始化完成!");
    let limg = image::open("images/left.png").unwrap().to_rgb8();
    let rimg = image::open("images/right.png").unwrap().to_rgb8();

    // 1. 源图处理
    let limg = image::imageops::blur(&limg, 1.0);
    let rimg = image::imageops::blur(&rimg, 1.0);

    let option = ADCensusOption::new(0, 64)
        .set_lrcheck_thres(1.0f32)
        .set_do_lr_check(true)
        .set_do_filling(true)
        .set_irv_th(0.4)
        .set_irv_ts(20.0)
        .set_do_discontinuity_adjustment(true)
        .build();

    let mut stereo = ad_census::ADCensus::new(limg.width(), limg.height(), Some(option)).unwrap();
    let mut sw = stopwatch::Stopwatch::start_new();
    let (disp_left, disp_right) = stereo.matching(&limg, &rimg).unwrap();
    info!("matching elapse time: {}ms", sw.elapsed_ms());
    sw.restart();
    utils::to_disparity_image(&disp_left)
        .unwrap()
        .save("display-left.png")
        .unwrap();
    utils::to_disparity_image(&disp_right)
        .unwrap()
        .save("display-right.png")
        .unwrap();
    debug!(
        "[match] save disparity image. elapse time: {}ms",
        sw.elapsed_ms()
    );
}
