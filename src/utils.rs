use image::{GrayImage, RgbImage};
use log::LevelFilter;
use log4rs::{
    append::console::ConsoleAppender,
    config::{Appender, Root},
    encode::pattern::PatternEncoder,
};
use ndarray::{parallel::prelude::*, prelude::*, Zip};

use ndarray_stats::QuantileExt;
use nshare::ToNdarray3;

pub fn init_log(level: &str) {
    let stdout: ConsoleAppender = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "{d(%Y-%m-%d %H:%M:%S)} [{h({l})}] - {m}{n}",
        )))
        .build();
    let log_config = log4rs::config::Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(
            Root::builder()
                .appender("stdout")
                .build(if level == "trace" {
                    LevelFilter::Trace
                } else if level == "debug" {
                    LevelFilter::Debug
                } else {
                    LevelFilter::Info
                }),
        )
        .unwrap();
    log4rs::init_config(log_config).unwrap();
}

#[allow(dead_code)]
/// 将二维数组转换为灰度图
pub fn to_grayimage(source: &RgbImage) -> Array2<u8> {
    let (width, height) = source.dimensions();
    let mut source_nd = source.clone().into_ndarray3();
    source_nd.swap_axes(0, 1);
    source_nd.swap_axes(1, 2);
    let mut ret = Array2::<u8>::zeros((height as usize, width as usize));
    Zip::indexed(&mut ret).par_for_each(|(y, x), val| {
        let pixel = source_nd.slice(s![y, x, ..]);
        let v = (pixel[2] as f32 * 0.299f32
            + pixel[1] as f32 * 0.587f32
            + pixel[0] as f32 * 0.114f32) as u8;
        *val = v
    });
    ret
}

/// 像素距离
pub fn pixel_distance(p1: &[u8], p2: &[u8]) -> i32 {
    (p1[0] as i32 - p2[0] as i32)
        .abs()
        .max((p1[1] as i32 - p2[1] as i32).abs())
        .max((p1[2] as i32 - p2[2] as i32).abs())
}

/// 计算汉明距离
pub fn hamming_distance(a: u8, b: u8) -> u8 {
    (a ^ b).count_ones() as u8
}

/// 转换为视差图
pub fn to_disparity_image(source: &ArrayView2<f32>) -> Option<GrayImage> {
    let (height, width) = source.dim();
    let mut abs_source = Array2::<f32>::zeros((height, width));

    par_azip!((
    r in &mut abs_source, s in source){
        if s.is_nan() {
            *r = 0f32;
        } else {
            *r = s.abs();
        }
    });
    let disparity_min = abs_source.min().unwrap();
    //let disparity_max = abs_source.max().unwrap();
    let d = abs_source.max().unwrap() - disparity_min;
    let mut ret = Array2::<u8>::zeros((height, width));
    par_azip!((
    r in &mut ret, s in abs_source.view()){
        let val = (s - *disparity_min) / d * 255f32;
        *r = val as u8;
    });
    GrayImage::from_raw(
        width as u32,
        height as u32,
        ret.as_slice().unwrap().to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::{hamming_distance, pixel_distance};

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(255u8, 1u8), 7);
    }

    #[test]
    fn test_pixel_distance() {
        let color1:[u8;3] = [54,8,238];
        let color2:[u8;3] = [29,29,32];
        assert_eq!(pixel_distance(&color1, &color2), 206);
    }
}
