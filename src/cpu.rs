extern crate image;

use std::sync::{Arc, Mutex};

use image::RgbImage;

use crate::core::{ADCensusOption, CrossArm, Point};
use crate::error::{self, Error, Result};
use crate::utils::{hamming_distance, pixel_distance};

use getset::{CopyGetters, Getters, MutGetters, Setters};
use ndarray::{parallel::prelude::*, prelude::*, Zip};
use ndarray_stats::QuantileExt;
use ndhistogram::{axis::Uniform, ndhistogram, Histogram};
use nshare::ToNdarray3;

use log::{debug, error, trace};

/// ## 根据十字交叉臂进行代价聚合
/// 代价聚合是将支持区的代价值累加，再除以支持区的像素数量，也就是计算支持区代价的均值，赋给中心像素的代价值
/// 同时，我们必须注意到的是，不同的聚合方向，支持区并不相同，即先水平再竖直的聚合方向和先竖直再水平的聚合方向，
/// 两者的支持区是不同的。所以需要分别用两种方向的数组来保存支持区的像素数。
///
/// **建议迭代 1 次**
///
///### 参数
/// * arms: &ArrayView2<CrossArm> 基于图像素像的十字交叉臂相关信息
/// * cost_array: &ArrayView3<f32> 视差聚合代价三维数组
///### 示例:
///```
/// let aggregate_cost = aggregate_cost_with_arm(&cross_arms.view(), &cost_array.view());
///```
///
fn aggregate_cost_with_arm(
    arms: &ArrayView2<CrossArm>,
    cost_array: &ArrayView3<f32>,
) -> Array3<f32> {
    let mut ret = Array3::<f32>::zeros(cost_array.dim());
    Zip::indexed(cost_array)
        .and(&mut ret)
        .par_for_each(|(y, x, d), cost, aggregate_cost| {
            let current_cross_arm = arms.get((y, x));
            let current_cross_arm = current_cross_arm.unwrap();
            // 横向臂范围
            let horizontal_arm_range = current_cross_arm.horizontal_arm_range();
            // 纵向臂范围
            let vertical_arm_range = current_cross_arm.vertical_arm_range();
            // 先横向再纵向
            // 当前像素横向支持区域像素点总数
            let area_count = current_cross_arm.pixel_count_in_area();
            // 当前像素横向臂 cost 合计(包括当前节点的值)
            let mut total = cost_array
                .slice(s![y, horizontal_arm_range.0..horizontal_arm_range.1, d])
                .sum();
            for p in vertical_arm_range.0..vertical_arm_range.1 {
                // 如果是当前节点, 略过
                if p == y {
                    continue;
                }
                let tmp_arm = arms.get((p, x));
                let tmp_arm = tmp_arm.unwrap();
                let tmp_arm_range = tmp_arm.horizontal_arm_range();
                total = total
                    + cost_array
                        .slice(s![p, tmp_arm_range.0..tmp_arm_range.1, d])
                        .sum();
            }
            //--------------------------
            // 先纵向再横向
            // 当前像素纵向臂 cost 合计(包括当前节点的值)
            total = total
                + cost_array
                    .slice(s![vertical_arm_range.0..vertical_arm_range.1, x, d])
                    .sum();
            for p in horizontal_arm_range.0..horizontal_arm_range.1 {
                // 如果是当前节点, 略过
                if p == x {
                    continue;
                }
                let tmp_arm = arms.get((y, p));
                let tmp_arm = tmp_arm.unwrap();
                let tmp_arm_range = tmp_arm.vertical_arm_range();
                total = total
                    + cost_array
                        .slice(s![tmp_arm_range.0..tmp_arm_range.1, p, d])
                        .sum();
            }
            // 支持区域数量 = 十字交叉臂支持区域数量 + 2
            // 2: 横向臂时当前像素本身 + 纵向臂时当前像素本身)
            *aggregate_cost = if area_count == 0 {
                *cost
            } else {
                (total - cost * 2f32) / area_count as f32
            };
        });
    ret
}

/// 边缘检测 - Sobel算子
/**
 * Sobel算子是一种用于边缘检测的离散微分算子，它结合了高斯平滑和微分求导。
 * 该算子用于计算图像明暗程度近似值，根据图像边缘旁边明暗程度把该区域内超过某个数的特定点记为边缘。
 * Sobel算子在Prewitt算子的基础上增加了权重的概念，认为相邻点的距离远近对当前像素点的影响是不同的，
 * 距离越近的像素点对应当前像素的影响越大，从而实现图像锐化并突出边缘轮廓。
 *
 * Sobel算子根据像素点上下、左右邻点灰度加权差，在边缘处达到极值这一现象检测边缘。
 * 对噪声具有平滑作用，提供较为精确的边缘方向信息。
 * 因为Sobel算子结合了高斯平滑和微分求导（分化），因此结果会具有更多的抗噪性，
 * 当对精度要求不是很高时，Sobel算子是一种较为常用的边缘检测方法。
 * Sobel算子 模板，以当前像素(p5)为中心 3x3 二维数组结构
 *      p1      p2      p3
 *      p4      p5      p6
 *      p7      p8      p9
 * 公式:
 *      gx = (p7 + 2*p8 + p9) - (p1 + 2p2 + p3)
 *      gy = (p3 + 2*p6 + p9) - (p1 + 2p4 + p7)
 */
fn edge_detect_with_sobel(source: &ArrayView2<f32>, threshold: f32) -> Array2<bool> {
    let (height, width) = source.dim();
    let mut ret = Array2::<bool>::default((height, width));
    Zip::indexed(&mut ret).par_for_each(|(y, x), ret_val| {
        if y == 0 || y == height - 1 || x == 0 || x == width - 1 {
            return;
        }
        // 以当前像素为中心做 3x3 的切片
        let p = source.slice(s![y - 1..y + 2, x - 1..x + 2]);
        let p: Vec<f32> = p
            .iter()
            .map(|x| if x.is_nan() { 0f32 } else { *x })
            .collect();
        let grad = ((p[6] + 2f32 * p[7] + p[8])
            - (p[0] + 2f32 * p[1] + p[2])
            - (p[2] + 2f32 * p[5] + p[8])
            - (p[0] + 2f32 * p[3] + p[6]))
            .abs();
        if grad > threshold {
            *ret_val = true;
        }
    });
    ret
}

#[derive(Getters, Setters, MutGetters, CopyGetters, Clone, Debug)]
pub struct ADCensus {
    /// 核线像对影像宽
    #[getset(get = "pub", set = "pub")]
    width: u32,
    /// 核线像对影像高
    #[getset(get = "pub", set = "pub")]
    height: u32,
    /// 算法参数
    #[getset(get = "pub", set = "pub")]
    option: ADCensusOption,
    /// 源视图 - 左图 3通道彩色数据 Rgb
    image_source_left: Array3<u8>,
    /// 源视图 - 右图 3通道彩色数据 Rgb
    image_source_right: Array3<u8>,
    /// 所有视差代价值
    cost_vals: Array3<f32>,
    /// 所有素质十字交叉臂
    cross_arms: Array2<CrossArm>,
    /// 左视差图
    disparity_left: Array2<f32>,
    /// 右视差图
    disparity_right: Array2<f32>,
}

impl Default for ADCensus {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            image_source_left: Array3::<u8>::default((0, 0, 0)),
            image_source_right: Array3::<u8>::default((0, 0, 0)),
            cost_vals: Array3::<f32>::default((0, 0, 0)),
            cross_arms: Array2::<CrossArm>::default((0, 0)),
            option: ADCensusOption::default(),
            disparity_left: Array2::<f32>::default((0, 0)),
            disparity_right: Array2::<f32>::default((0, 0)),
        }
    }
}

impl ADCensus {
    pub fn new(width: u32, height: u32, option: Option<ADCensusOption>) -> Result<Self> {
        if width <= 0 || height <= 0 {
            return Err(Error::new(1001, error::ERROR_1001));
        }
        let mut ret = Self::default().set_width(width).set_height(height).build();
        if let Some(v) = option {
            ret.set_option(v);
        }
        // 视差范围必须大于0
        if ret.get_disparity_range() <= 0 {
            return Err(Error::new(1002, error::ERROR_1002));
        }
        Ok(ret)
    }
    ///将源图转换成灰度图
    fn source_image_to_gray(&self) -> (Array2<u8>, Array2<u8>) {
        let mut ret_left = Array2::<u8>::zeros((self.height as usize, self.width as usize));
        let mut ret_right = Array2::<u8>::zeros((self.height as usize, self.width as usize));
        Zip::indexed(&mut ret_left)
            .and(&mut ret_right)
            .par_for_each(|(y, x), val_left, val_right| {
                // 左图像素
                // pixel[0] = r
                // pixel[1] = g
                // pixel[2] = b
                let mut pixel = self.image_source_left.slice(s![y, x, ..]);
                let mut v = (pixel[0] as f32 * 0.299f32
                    + pixel[1] as f32 * 0.587f32
                    + pixel[2] as f32 * 0.114f32) as u8;
                *val_left = v;
                // 右图像素
                pixel = self.image_source_right.slice(s![y, x, ..]);
                v = (pixel[0] as f32 * 0.299f32
                    + pixel[1] as f32 * 0.587f32
                    + pixel[2] as f32 * 0.114f32) as u8;
                *val_right = v;
            });
        (ret_left, ret_right)
    }
    /// ## Census变换
    ///
    /// ### 参数
    ///  * neighborhood_window_width: 邻域窗口宽度
    ///  * neighborhood_window_height: 邻域窗口高度
    /// neighborhood_window_width 和 neighborhood_window_height 必须为奇数
    ///
    /*
    在视图中选取任一点，以该点为中心划出一个例如3 × 3 的矩形，
    矩形中除中心点之外的每一点都与中心点进行比较，灰度值小于中心点记为1，灰度大于中心点的则记为0，
    以所得长度为 8 的只有 0 和 1 的序列作为该中心点的 census 序列,即中心像素的灰度值被census 序列替换。
    具体而言，对于欲求取视差的左右视图，要比较两个视图中两点的相似度，可将此两点的census值逐位进行异或运算，
    然后计算结果为1 的个数，记为此两点之间的汉明值，汉明值是两点间相似度的一种体现，
    汉明值愈小，两点相似度愈大实现算法时先异或再统计1的个数即可，汉明距越小即相似度越高。

    127  126  130                1    1    0
    126  128  129      --->      1    *    0    ---> cencus序列 ｛11010101｝
    127  131  111                1    0    1
                                                            --->  异或 ｛01110010｝ --->汉明距为4
    110  126  101                1    0    1
    146  120  127      --->      0    *    0    ---> cencus序列  ｛10100111｝
    112  101  111                1    1    1
     */
    fn census_transform(
        &self,
        neighborhood_window_width: u8,
        neighborhood_window_height: u8,
    ) -> Result<(Array2<u8>, Array2<u8>)> {
        if (neighborhood_window_width % 2 == 0) || (neighborhood_window_height % 2 == 0) {
            return Err(Error::new(1003, error::ERROR_1003));
        }
        let (imgy, imgx, _) = self.image_source_left.dim();
        let (image_gray_left, image_gray_right) = self.source_image_to_gray();
        // trace!(
        //     "    [census_transform] begin, image info: {} x {}, neighborhood_window: {} x {}",
        //     imgx,
        //     imgy,
        //     neighborhood_window_width,
        //     neighborhood_window_height,
        // );
        let start_x = ((neighborhood_window_width - 1) / 2) as usize;
        let start_y = ((neighborhood_window_height - 1) / 2) as usize;
        // 处理左图
        let mut census_nd_left = Array2::<u8>::zeros((imgy, imgx));
        // 处理右图
        let mut census_nd_right = Array2::<u8>::zeros((imgy, imgx));
        Zip::indexed(&image_gray_left)
            .and(&image_gray_right)
            .and(&mut census_nd_left)
            .and(&mut census_nd_right)
            .par_for_each(
                |(y, x), img_left_p, img_right_p, census_left_val, census_right_val| {
                    if x < start_x || x >= imgx - start_x {
                        return;
                    }
                    if y < start_y || y >= imgy - start_y {
                        return;
                    }
                    let mut val_l: u8 = 0;
                    let mut val_r: u8 = 0;
                    for dx in (0 - start_x as isize)..start_x as isize {
                        for dy in (0 - start_y as isize)..start_y as isize {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            val_l = val_l << 1;
                            val_r = val_r << 1;
                            let xdx = x as isize + dx;
                            let ydy = y as isize + dy;
                            let tdata_l =
                                image_gray_left.get((ydy as usize, xdx as usize)).unwrap();
                            let tdata_r =
                                image_gray_left.get((ydy as usize, xdx as usize)).unwrap();
                            val_l = val_l + if tdata_l < img_left_p { 0 } else { 1 };
                            val_r = val_r + if tdata_r < img_right_p { 0 } else { 1 };
                        }
                    }
                    // 左图中心像素的census值
                    *census_left_val = val_l;
                    // 右图中心像素的census值
                    *census_right_val = val_r;
                },
            );
        //trace!("    [census_transform] compute census end. ");
        Ok((census_nd_left, census_nd_right))
    }
    /// 计算代价
    fn compute_cost(&mut self, left_census: &ArrayView2<u8>, right_census: &ArrayView2<u8>) {
        // 获取视差范围
        let min_disparity = *self.option.min_disparity();
        let max_disparity = *self.option.max_disparity();
        // 预设参数
        let lambda_ad = *self.option.lambda_ad() as f32;
        let lambda_census = *self.option.lambda_census() as f32;
        self.cost_vals =
            Array3::<f32>::zeros((*self.height() as usize, *self.width() as usize, 0usize));

        for drange in min_disparity..max_disparity {
            let mut ret = Array2::<f32>::ones((*self.height() as usize, *self.width() as usize));
            Zip::indexed(&mut ret)
                .and(left_census)
                .and(right_census)
                .par_for_each(|(y, x), cost, census_val_l, census_val_r| {
                    let xr = x as i32 - drange;
                    if xr < 0 || xr >= *self.width() as i32 {
                        *cost = 1f32;
                        return;
                    }
                    // 左图像素 RGB
                    let pixel_left = self.image_source_left.slice(s![y, x, ..]);
                    // 右图像素 RGB
                    let pixel_right = self.image_source_right.slice(s![y, xr, ..]);
                    let cost_ad = ((pixel_left[0] as f32 - pixel_right[0] as f32).abs()
                        + (pixel_left[1] as f32 - pixel_right[1] as f32).abs()
                        + (pixel_left[2] as f32 - pixel_right[2] as f32).abs())
                        / 3f32;
                    let cost_census = hamming_distance(*census_val_l, *census_val_r) as f32;
                    *cost = 1f32 - ((0f32 - cost_ad) / lambda_ad).exp() + 1f32
                        - ((0f32 - cost_census) / lambda_census).exp();
                });
            let _ = self.cost_vals.push(Axis(2), ret.view());
        }
    }
    /// 构建左图十字交叉臂
    /// 1. 计算每个像素的(左/右/上/下)臂长
    /// 2. 计算每个像素十字交叉臂支持区域像素数量, 计算分为:
    ///     先横向再纵向计算, 数量保存在 CrossArm.pixel_count_in_area_by_horizontal 属性中
    ///     先纵向再横向计算, 数量保存在 CrossArm.pixel_count_in_area_by_vertical 属性中
    /// 3. 标记每个像素
    ///     横向臂(左臂+右臂) (起始像素—x, 结束像素-x), 数据保存在 CrossArm.horizontal_arm_range 属性中
    ///     纵向臂(上臂+下臂) (起始像素—y, 结束像素-y), 数据保存在 CrossArm.vertical_arm_range 属性中
    ///
    ///
    fn build_cross_arm(&self) -> Array2<CrossArm> {
        // 构建十字交叉臂步骤
        // 1. 遍历整张左图象素
        // 2. 根据当前像素位置（即:中心像素），臂长阀值进行以行(左右切)和列(上下切)的切片
        // 3. 遍历切片，判断臂长，规则如下
        //      1. 切片遍历时当前像素和方向上前一个像素的色差都小于t1
        //      2. 切片遍历的像素与中心像素的距离 > L2时，色差必须小于t2, 公式: 中心像素位置 - 当前像素 > L2
        //              a. 横向时，用x;纵向时，用y
        //      3. 切片数量不能大于L1
        let (img_height, img_width) = (self.height as usize, self.width as usize);
        let cross_l1 = *self.option().cross_l1() as i32;
        let cross_l2 = *self.option().cross_l2() as i32;
        let cross_t1 = *self.option().cross_t1();
        let cross_t2 = *self.option().cross_t2();
        trace!(
            "    [build_cross_arm] image info: {} x {}, L1:{}, L2:{}, t1:{}, t2:{}",
            img_width,
            img_height,
            cross_l1,
            cross_l2,
            cross_t1,
            cross_t2
        );

        let mut ret = Array2::<CrossArm>::default((img_height, img_width));
        // 构建十字交叉臂(获取横向臂/纵向臂信息)
        Zip::indexed(&mut ret).par_for_each(|(y, x), cross_arm| {
            let pos_x = x as i32;
            let pos_y = y as i32;
            // 水平向左延伸臂长参数(起始点, 结束点), 不含当前像素
            let dir_left = if pos_x - cross_l1 <= 0 {
                (0, if pos_x == 0 { 0 } else { pos_x - 1 })
            } else {
                (pos_x - cross_l1, pos_x - 1)
            };
            // 水平向右延伸臂长参数(起始点, 结束点), 不含当前像素
            let dir_right = if pos_x + cross_l1 >= img_width as i32 {
                (pos_x + 1, img_width as i32)
            } else {
                (pos_x + 1, pos_x + cross_l1)
            };
            // 纵向向上延伸臂长参数(起始点, 结束点), 不含当前像素
            let dir_top = if pos_y - cross_l1 <= 0 {
                (0, if pos_y == 0 { 0 } else { pos_y - 1 })
            } else {
                (pos_y - cross_l1, pos_y - 1)
            };
            // 水平向右延伸臂长参数(起始点, 结束点), 不含当前像素
            let dir_bottom = if pos_y + cross_l1 >= img_height as i32 {
                (pos_y + 1, img_height as i32 - 1)
            } else {
                (pos_y + 1, pos_y + cross_l1)
            };
            let pixel_color_center = self.image_source_left.slice(s![y, x, ..]);
            let pixel_color_center = pixel_color_center.as_slice().unwrap();

            //开始横向
            //左边
            let mut val = 0u8;
            for i in (dir_left.0..dir_left.1).rev() {
                //根据当前像素与中心点距离, 获取色差阈值
                let current_t = if (x as i32 - i) > cross_l2 {
                    cross_t2
                } else {
                    cross_t1
                };
                // 当前像素与中心点的色差
                let pixel_color_current = self.image_source_left.slice(s![y, i, ..]);
                let pixel_color_current = pixel_color_current.as_slice().unwrap();
                if pixel_distance(pixel_color_center, pixel_color_current) < current_t {
                    // 获取下一个像素与中心点的色差
                    if i - 1 > 0 {
                        let next_d = {
                            let pixel_color_next = self.image_source_left.slice(s![y, i - 1, ..]);
                            pixel_distance(pixel_color_center, pixel_color_next.as_slice().unwrap())
                        };
                        // 如果下一个像素的色差 大于 色差阈值，说明当前像素为臂边缘像素
                        if next_d > current_t {
                            break;
                        }
                    }
                    val = val + 1;
                } else {
                    break;
                }
            }
            cross_arm.set_left(val);
            //右边
            val = 0u8;
            for i in dir_right.0..dir_right.1 {
                let current_t = if (i - x as i32) > cross_l2 {
                    cross_t2
                } else {
                    cross_t1
                };
                // 当前像素与中心点的色差
                let pixel_color_current = self.image_source_left.slice(s![y, i, ..]);
                let pixel_color_current = pixel_color_current.as_slice().unwrap();
                if pixel_distance(pixel_color_center, pixel_color_current) < current_t {
                    if i < (self.width - 1) as i32 {
                        let next_d = {
                            let pixel_color_next = self.image_source_left.slice(s![y, i + 1, ..]);
                            pixel_distance(pixel_color_center, pixel_color_next.as_slice().unwrap())
                        };
                        // 如果下一个像素的色差 大于 色差阈值，说明当前像素为臂边缘像素
                        if next_d > current_t {
                            break;
                        }
                    }
                    val = val + 1;
                } else {
                    break;
                }
            }
            cross_arm.set_right(val);
            //---------------------
            //开始纵向
            // 上边
            val = 0u8;
            for i in (dir_top.0..dir_top.1).rev() {
                //根据当前像素与中心点距离, 获取色差阈值
                let current_t = if (y as i32 - i) > cross_l2 {
                    cross_t2
                } else {
                    cross_t1
                };
                // 当前像素与中心点的色差
                let pixel_color_current = self.image_source_left.slice(s![i, x, ..]);
                let pixel_color_current = pixel_color_current.as_slice().unwrap();
                if pixel_distance(pixel_color_center, pixel_color_current) < current_t {
                    // 获取下一个像素与中心点的色差
                    if i - 1 > 0 {
                        let next_d = {
                            let pixel_color_next = self.image_source_left.slice(s![i - 1, x, ..]);
                            pixel_distance(pixel_color_center, pixel_color_next.as_slice().unwrap())
                        };
                        // 如果下一个像素的色差 大于 色差阈值，说明当前像素为臂边缘像素
                        if next_d > current_t {
                            break;
                        }
                    }
                    val = val + 1;
                } else {
                    break;
                }
            }
            cross_arm.set_top(val);
            // 下边
            val = 0u8;
            for i in dir_bottom.0..dir_bottom.1 {
                let current_t = if (i - y as i32) > cross_l2 {
                    cross_t2
                } else {
                    cross_t1
                };
                // 当前像素与中心点的色差
                let pixel_color_current = self.image_source_left.slice(s![i, x, ..]);
                let pixel_color_current = pixel_color_current.as_slice().unwrap();
                if pixel_distance(pixel_color_center, pixel_color_current) < current_t {
                    if i + 1 < (self.width - 1) as i32 {
                        let next_d = {
                            let pixel_color_next = self.image_source_left.slice(s![i + 1, x, ..]);
                            pixel_distance(pixel_color_center, pixel_color_next.as_slice().unwrap())
                        };
                        if next_d > current_t {
                            break;
                        }
                    }
                    val = val + 1;
                } else {
                    break;
                }
            }
            cross_arm.set_bottom(val);
            //设置横向臂起止范围, 供统计支持区像素时使用
            let start_x = x - *cross_arm.left() as usize;
            let start_y = y - *cross_arm.top() as usize;
            let end_x = x + *cross_arm.right() as usize;
            let end_y = y + *cross_arm.bottom() as usize;
            cross_arm.set_horizontal_arm_range((start_x, end_x));
            //设置纵向臂起止范围, 供统计支持区像素时使用
            cross_arm.set_vertical_arm_range((start_y, end_y));
            //--------------------
        });
        //统计每次像素的支持区（十字交叉臂覆盖的区域）像素的数量
        let ret_view = ret.clone();
        Zip::indexed(&mut ret).par_for_each(|(y, x), cross_arm| {
            let (start_y, end_y) = cross_arm.vertical_arm_range();
            // 横向统计区域数量
            let mut total_count = cross_arm.horizontal_pixel_count();
            for current_y in *start_y..*end_y {
                if current_y == y {
                    continue;
                }
                if let Some(tmp_arm) = ret_view.get((current_y, x)) {
                    total_count = total_count + tmp_arm.vertical_pixel_count();
                }
            }
            cross_arm.set_pixel_count_in_area_by_horizontal(total_count);
            // 纵向统计区域数量
            let (start_x, end_x) = cross_arm.horizontal_arm_range();
            total_count = cross_arm.vertical_pixel_count();
            for current_x in *start_x..*end_x {
                if current_x == x {
                    continue;
                }
                if let Some(tmp_arm) = ret_view.get((current_x, y)) {
                    total_count = total_count + tmp_arm.horizontal_pixel_count();
                }
            }
            cross_arm.set_pixel_count_in_area_by_vertical(total_count);
        });
        ret
    }
    /// 代价聚合
    fn aggregate_cost(&mut self, iter_times: u8) {
        // 构建像素的十字交叉臂
        self.cross_arms = self.build_cross_arm();
        trace!(
            "    [aggregate_cost] build cross arm end. cross_arm info : {:?}",
            self.cross_arms.dim()
        );

        trace!("    [aggregate_cost] begin to aggregate cost... ");
        let mut ret = aggregate_cost_with_arm(&self.cross_arms.view(), &self.cost_vals.view());
        //运行迭代次数
        //在上一次的基础之上再做一次代价聚合
        for _ in 0..iter_times {
            ret = aggregate_cost_with_arm(&self.cross_arms.view(), &ret.view());
        }
        trace!("    [aggregate_cost] aggregate cost end. ");
        self.cost_vals = ret;
    }
    /// 扫描线优化
    /// 1. 从左往右
    /// 2. 从右往左
    /// 3. 从上往下
    /// 4. 从下往上
    fn scanline_optimize(&mut self) {
        // 从左往右扫描
        let mut r = self.scanline_optimize_from_left(
            &self.image_source_left.view(),
            &self.image_source_right.view(),
            &self.cost_vals.view(),
        );
        /* 从右往左扫描
         * 1. 将左图水平翻转
         * 2. 将右图水平翻转
         * 3. 将聚合代价以 列轴 进行水平翻转
         * 4. 从左往右进行扫描线优化
         * 5. 将结果以 列轴 进行水平翻转
         */
        let mut tmp_left_image = self.image_source_left.slice(s!(.., ..;-1, ..));
        let mut tmp_right_image = self.image_source_right.slice(s!(.., ..;-1, ..));
        let mut tmp_aggregate_cost = r.slice(s!(.., ..;-1, ..));
        r = self.scanline_optimize_from_left(
            &tmp_left_image,
            &tmp_right_image,
            &tmp_aggregate_cost,
        );
        r = r.slice(s!(.., ..;-1, ..)).to_owned();
        // ----------------------------------------
        /* 从上往下扫描
         * 1. 将左图 x轴与y轴 交换
         * 2. 将右图 x轴与y轴 交换
         * 3. 将聚合代价 x轴与y轴 交换
         * 4. 从左往右进行扫描线优化
         * 5. 将结果还原 x轴与y轴 交换
         */
        // 处理左图
        tmp_left_image = self.image_source_left.view();
        tmp_left_image.swap_axes(0, 1);
        // 处理右图
        tmp_right_image = self.image_source_right.view();
        tmp_right_image.swap_axes(0, 1);
        // 处理聚合代价
        tmp_aggregate_cost = r.view();
        tmp_aggregate_cost.swap_axes(0, 1);

        r = self.scanline_optimize_from_left(
            &tmp_left_image,
            &tmp_right_image,
            &tmp_aggregate_cost,
        );
        // 处理结果
        r.swap_axes(0, 1);
        // ----------------------------------------
        /* 从下往上扫描
         * 1. 将左图 x轴与y轴 交换, 再水平翻转
         * 2. 将右图 x轴与y轴 交换, 再水平翻转
         * 3. 将聚合代价 x轴与y轴 交换, 再水平翻转
         * 4. 从左往右进行扫描线优化
         * 5. 将结果还原 水平翻转 x轴与y轴 交换
         */
        // 处理左图
        tmp_left_image = self.image_source_left.view();
        tmp_left_image.swap_axes(0, 1);
        let tmp_left_image = tmp_left_image.slice(s!(.., ..;-1, ..));
        // 处理右图
        tmp_right_image = self.image_source_right.view();
        tmp_right_image.swap_axes(0, 1);
        let tmp_right_image = tmp_right_image.slice(s!(.., ..;-1, ..));
        // 处理聚合代价
        tmp_aggregate_cost = r.view();
        tmp_aggregate_cost.swap_axes(0, 1);
        let tmp_aggregate_cost = tmp_aggregate_cost.slice(s!(.., ..;-1, ..));
        r = self.scanline_optimize_from_left(
            &tmp_left_image,
            &tmp_right_image,
            &tmp_aggregate_cost,
        );
        let mut ret = r.slice(s!(.., ..;-1, ..));
        ret.swap_axes(0, 1);
        // ----------------------------------------
        self.cost_vals = ret.to_owned();
    }
    /// 扫描线优化(从左往右)
    fn scanline_optimize_from_left(
        &self,
        left_image: &ArrayView3<u8>,
        right_image: &ArrayView3<u8>,
        aggregate_cost: &ArrayView3<f32>,
    ) -> Array3<f32> {
        let so_p1 = *self.option().so_p1();
        let so_p2 = *self.option().so_p2();
        let so_tso = *self.option().so_p2() as i32;
        // 计算结果
        let (height, width, depth) = aggregate_cost.dim();
        let mut ret = Array3::<f32>::zeros((height, width, depth));
        // 从左 -> 右路径
        Zip::indexed(aggregate_cost)
            .and(&mut ret)
            .par_for_each(|(y, w, d), cost, ret_val| {
                let x = w;
                // 首列像素跳过
                if x == 0 || x >= width - 1 {
                    return;
                }
                // 左视图相邻像素色差

                let mut pixel = left_image.slice(s![y, x, ..]);
                // 如果为首列时，首列的上一像素为0
                // 左 -> 右时, 为 x-1
                // 右 -> 左时, 为 x+1
                let mut pixel_prev = left_image.slice(s![y, x - 1, ..]);
                let d1 = pixel_distance(pixel.as_slice().unwrap(), pixel_prev.as_slice().unwrap());

                let mut d2 = d1;
                // -------------------
                // 右视图相邻像素色差
                let disparity_x = x as i32 - d as i32 - *self.option.min_disparity();
                if disparity_x > 0 && disparity_x < width as i32 - 1 {
                    pixel = right_image.slice(s![y, disparity_x as usize, ..]);
                    pixel_prev = right_image.slice(s![y, (disparity_x - 1) as usize, ..]);
                    d2 = pixel_distance(pixel.as_slice().unwrap(), pixel_prev.as_slice().unwrap());
                }

                // -------------------
                // 获取P1和P2
                let (p1, p2) = if d1 >= so_tso && d2 >= so_tso {
                    (so_p1 / 10f32, so_p2 / 10f32)
                } else if d1 < so_tso && d2 >= so_tso {
                    (so_p1 / 4f32, so_p2 / 4f32)
                } else if d1 >= so_tso && d2 < so_tso {
                    (so_p1 / 4f32, so_p2 / 4f32)
                } else {
                    (so_p1, so_p2)
                };
                // -------------
                // 当前像素在当前视差下，上一个像素的列坐标
                // 左 -> 右时, 为 x-1
                // 右 -> 左时, 为 x+1
                let last_x = x - 1;
                // l1: 当前像素的前一个像素在当前视差下(d)的聚合代价
                let l1 = *aggregate_cost.get((y, last_x, d)).unwrap();
                // l2: 当前像素的前一个像素在当前视差的前一个视差(d-1)的聚合代价
                let l2 = *aggregate_cost.get((y, last_x, d - 1)).unwrap_or(cost) + p1;
                // l2: 当前像素的前一个像素在当前视差的后一个视差(d+1)的聚合代价
                let l3 = *aggregate_cost.get((y, last_x, d + 1)).unwrap_or(cost) + p1;
                // 方向上上一个节点所有视差图下最小聚合代价值
                let mincost_last_path = match aggregate_cost.slice(s![y, last_x, 0..d]).min() {
                    Ok(v) => *v,
                    Err(_e) => {
                        //trace!("({}, {}, {}): error: {:?}", y, x, d, e);
                        *cost
                    }
                };
                let l4 = mincost_last_path + p2;
                // Lr(p,d) = C(p,d)
                //    + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 )
                //    - min(Lr(p-r))
                let v = (*cost + l1.min(l2).min(l3).min(l4)) / 2f32;
                *ret_val = v
            });
        ret
    }
    /// 计算视差
    /// 通过代价聚合计算左右图视差
    fn compute_disparity(&mut self) {
        let mut disparity_left = Array2::<f32>::zeros((self.height as usize, self.width as usize));
        let mut disparity_right = Array2::<f32>::zeros((self.height as usize, self.width as usize));
        Zip::indexed(&mut disparity_left)
            .and(&mut disparity_right)
            .par_for_each(|(y, x), ret_left_val, ret_right_val| {
                // 获取当前像素所有视差下代价值
                let costs = self.cost_vals.slice(s![y, x, ..]);
                // 当前像素左视差及深度位置
                let (disparity_left_val, disparity_right_val) = match costs.argmin() {
                    Ok(depth_left) => {
                        if depth_left == 0 || depth_left == costs.len() - 1 {
                            (f32::NAN, depth_left as f32)
                        } else {
                            // 左视差
                            let min_cost = costs[depth_left];
                            // 最优视差前一个视差的代价值min_cost_up, 后一个视差的代价值min_cost_down
                            let mut min_cost_up = costs[depth_left - 1];
                            let mut min_cost_down = costs[depth_left + 1];
                            // 解一元二次曲线极值
                            let mut denom = min_cost_up + min_cost_down - 2f32 * min_cost;
                            let left_val = if denom != 0f32 {
                                depth_left as f32 + (min_cost_up - min_cost_down) / (denom * 2f32)
                            } else {
                                depth_left as f32
                            };
                            // 右视差
                            // 通过左影像的代价，获取右影像的代价
                            // 右cost(xr,yr,d) = 左cost(xr+d,yl,d)
                            let depth_right = depth_left + x;
                            if depth_right > 0 && depth_right < costs.len() - 1 {
                                //min_cost = costs[depth_right];
                                min_cost_up = costs[depth_right - 1];
                                min_cost_down = costs[depth_right + 1];
                            }
                            // 解一元二次曲线极值
                            denom = min_cost_up + min_cost_down - 2f32 * min_cost;
                            let right_val = if denom != 0f32 {
                                depth_left as f32 + (min_cost_up - min_cost_down) / (denom * 2f32)
                            } else {
                                depth_left as f32
                            };
                            (left_val, right_val)
                        }
                    }
                    Err(_e) => {
                        //trace!("[compute_disparity]: ({}, {}), error: {:?}", y, x, e);
                        (costs[0], 0f32)
                    }
                };
                *ret_left_val = disparity_left_val;
                *ret_right_val = disparity_right_val;
            });
        self.disparity_left = disparity_left;
        self.disparity_right = disparity_right;
    }
    /// 立体匹配
    pub fn matching(
        &mut self,
        left_image: &RgbImage,
        right_image: &RgbImage,
    ) -> Result<(ArrayView2<f32>, ArrayView2<f32>)> {
        // 立体匹配的图像不能为空
        if left_image.width() * left_image.height() <= 0
            || right_image.width() * right_image.height() <= 0
        {
            return Err(Error::new(1001, error::ERROR_1001));
        }
        let mut sw = stopwatch::Stopwatch::start_new();
        //debug!("[match] image blur end. elapse time: {}ms", sw.elapsed_ms());
        self.set_width(left_image.width());
        self.set_height(left_image.height());
        // 左图转换为数组(RGB)
        self.image_source_left = left_image.clone().into_ndarray3();
        self.image_source_left.swap_axes(0, 1);
        self.image_source_left.swap_axes(1, 2);
        // 右图转换为数组
        self.image_source_right = right_image.clone().into_ndarray3();
        self.image_source_right.swap_axes(0, 1);
        self.image_source_right.swap_axes(1, 2);
        debug!(
            "[match] image to ndarray. left source(H x W x C): {:?}, right source: {:?}, elapse time: {}ms",
            self.image_source_left.dim(),
            self.image_source_right.dim(),
            sw.elapsed_ms()
        );
        // 2. Census变换 9x7
        sw.restart();
        let (left_census, right_census) = self.census_transform(9, 7)?;
        //trace!("[match] left_census: {:?}\n, right_census: {:?}", left_census, right_census);
        debug!(
            "[match] compute census end. elapse time: {}ms, value: {} - {}",
            sw.elapsed_ms(),
            left_census.min().unwrap(),
            left_census.max().unwrap()
        );

        // 3. 代价计算
        sw.restart();
        debug!(
            "[match] compute cost begin, left census: {:?}, right census: {:?}",
            left_census.dim(),
            right_census.dim()
        );
        self.compute_cost(&left_census.view(), &right_census.view());
        //trace!("cost vals: {:?}", &self.cost_vals.view());
        debug!(
            "[match] compute cost end. value: {} - {}. elapse time: {}ms",
            self.cost_vals.min().unwrap(),
            self.cost_vals.max().unwrap(),
            sw.elapsed_ms(),
        );
        // self.compute_disparity();
        // to_disparity_image(&self.disparity_left.view())
        //     .unwrap()
        //     .save("display-left.png")
        //     .unwrap();
        // 计算代价完成.---------------
        // 4. 代价聚合(效率较低)
        sw.restart();
        self.aggregate_cost(1u8);

        // self.compute_disparity();
        // to_disparity_image(&self.disparity_left.view())
        //     .unwrap()
        //     .save("display-left.png")
        //     .unwrap();

        debug!(
            "[match] aggregate cost end. , value: {} - {}. elapse time: {}ms",
            self.cost_vals.min().unwrap(),
            self.cost_vals.max().unwrap(),
            sw.elapsed_ms()
        );

        // 5. 扫描线优化
        sw.restart();
        self.scanline_optimize();
        // self.compute_disparity();
        // to_disparity_image(&self.disparity_left.view())
        //     .unwrap()
        //     .save("display_left.png")
        //     .unwrap();
        debug!(
            "[match] scanline_optimize end. elapse time: {}ms, {:?}",
            sw.elapsed_ms(),
            self.cost_vals.dim()
        );

        // 6. 计算左右视图视差
        sw.restart();
        self.compute_disparity();
        debug!(
            "[match] compute left/right disparity. elapse time: {}ms",
            sw.elapsed_ms()
        );
        // 7. 多步骤视差优化
        sw.restart();
        self.multistep_refiner();
        debug!(
            "[match] multistep refiner. elapse time: {}ms",
            sw.elapsed_ms()
        );
        Ok((self.disparity_left.view(), self.disparity_right.view()))
    }
    /// 多步骤优化器
    fn multistep_refiner(&mut self) {
        // 1. 离群点检测
        let mut mismatches = Array1::<Point>::default(0);
        let mut occlusions = Array1::<Point>::default(0);
        if self.option().do_lr_check() == &true {
            let (vec_mismatches, vec_occlusions) = self.lr_check();
            mismatches = Array1::from_vec(vec_mismatches);
            occlusions = Array1::from_vec(vec_occlusions);
            trace!(
                "    [multistep_refiner] lr-check end. mismatches({}) occlusions({}).",
                mismatches.dim(),
                occlusions.dim()
            );
        }
        // 2. 迭代局部投票
        if self.option().do_filling() == &true {
            self.iterative_region_voting(&mut mismatches.view_mut(), &mut occlusions.view_mut());
            // 过滤掉已处理的误匹配像素(像素坐标(0, 0))
            mismatches = Array1::from_vec(
                mismatches
                    .into_par_iter()
                    // 过滤像素坐标为 (0, 0)
                    .filter(|p| p.x() > &0 && p.y() > &0)
                    .map(|p| *p)
                    .collect::<Vec<_>>(),
            );
            // 过滤掉已处理的遮挡像素(像素坐标(0, 0))
            occlusions = Array1::from_vec(
                occlusions
                    .into_par_iter()
                    // 过滤像素坐标为 (0, 0)
                    .filter(|p| p.x() > &0 && p.y() > &0)
                    .map(|p| *p)
                    .collect::<Vec<_>>(),
            );
            trace!(
                "    [multistep_refiner] iterative region voting end. mismatches({}) occlusions({}).",
                mismatches.dim(),
                occlusions.dim()
            );
            // 3. 内插填充
            self.proper_interpolation(&mut mismatches.view_mut(), &mut occlusions.view_mut());
            trace!("    [multistep_refiner] proper interpolation end. ");
        }
        // 4. 深度非连续区视差调整
        if self.option().do_discontinuity_adjustment() == &true {
            self.depth_discontinuity_adjustment();
            trace!("    [multistep_refiner] depth discontinuity adjustment end. ");
        }
        // 5. 中值滤波
    }
    /// 深度非连续区视差调整
    fn depth_discontinuity_adjustment(&mut self) {
        let disp_left = self.disparity_left.clone();
        let edge = edge_detect_with_sobel(&disp_left.view(), 5.0f32);
        let (_, width) = disp_left.dim();
        let costs = self.cost_vals.view();
        Zip::indexed(&edge)
            .and(&mut self.disparity_left.view_mut())
            .par_for_each(|(y, x), edge_val, disp_val| {
                if edge_val == &false || disp_val.is_nan() == true || x < 1 || x > width - 1 {
                    return;
                }
                let d = disp_val.round() as usize;
                let cost = costs.get((y, x, d)).unwrap();
                // 记录左右两边像素的视差值和代价值
                // 选择代价最小的像素视差值
                for k in 0..2 {
                    let xx = if k == 0 { x - 1 } else { x + 1 };
                    let dd = disp_left.get((y, xx)).unwrap();
                    if dd.is_nan() {
                        continue;
                    }
                    let tmp_depth = dd.round() as usize;
                    let cc = *costs.get((y, xx, tmp_depth)).unwrap();
                    if *cost > cc {
                        //*cost = cc;
                        *disp_val = *dd;
                    }
                }
            });
    }
    /// 内插填充
    fn proper_interpolation(
        &mut self,
        mismatches: &mut ArrayViewMut1<Point>,
        occlusions: &mut ArrayViewMut1<Point>,
    ) {
        let pi = 3.1415926f32;
        let width = *self.width() as usize;
        let height = *self.height() as usize;
        let disp_max = *self.option().cross_l1() as u32;
        for k in 0..2 {
            let process_pixels = if k == 0 {
                mismatches.view_mut()
            } else {
                occlusions.view_mut()
            };
            let mut valid_points: Vec<(f32, Point)> = Vec::new();
            let mut valid_points_val: Vec<f32> = Vec::new();
            let mut ang = 0f32;
            Zip::indexed(process_pixels).for_each(|_, p| {
                let pos_y = *p.y();
                let pos_x = *p.x();
                // 计算16个方向上的遇到的首个有效视差值(最多16个值)
                ang = 0f32;
                valid_points.clear();
                valid_points_val.clear();
                for _ in 0..16 {
                    let sina = ang.sin();
                    let cosa = ang.cos();
                    for i in 0..disp_max {
                        let yy = (pos_y as f32 + i as f32 * sina).round() as usize;
                        let xx = (pos_x as f32 + i as f32 * cosa).round() as usize;
                        if yy >= height || xx >= width {
                            break;
                        }
                        match self.disparity_left.get((yy, xx)) {
                            Some(v) => {
                                if v.is_nan() == false && !valid_points_val.contains(v) {
                                    valid_points_val.push(*v);
                                    valid_points.push((*v, Point::new(xx, yy)));
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    ang += pi / 16f32;
                }
                if valid_points.is_empty() {
                    return;
                } else {
                    //排序从小到大
                    valid_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }

                // 如果是误匹配区，则选择颜色最相近的像素视差值
                // 如果是遮挡区，则选择最小视差值
                let d = self.disparity_left.get_mut((pos_y, pos_x)).unwrap();
                if k == 0 {
                    let pixel_color_target = self.image_source_left.slice(s![pos_y, pos_x, ..]);
                    let pixel_color_target = pixel_color_target.as_slice().unwrap();
                    let mut min_dist = 999f32;
                    for i in 0..valid_points.len() {
                        let (d, p) = valid_points[i];
                        let pixel_color_current =
                            self.image_source_left.slice(s![*p.y(), *p.x(), ..]);
                        let pixel_color_current = pixel_color_current.as_slice().unwrap();
                        if ((pixel_distance(pixel_color_target, pixel_color_current)) as f32)
                            < min_dist
                        {
                            min_dist = d;
                        }
                    }
                    *d = min_dist as f32;
                } else {
                    *d = valid_points_val[0];
                }
                //p.set_x(0);
                //p.set_y(0);
            });
        }
    }
    /// 迭代局部投票
    /**
     * 对处理过的像素, 将其坐标标记为(0, 0)
     */
    fn iterative_region_voting(
        &mut self,
        mismatches: &mut ArrayViewMut1<Point>,
        occlusions: &mut ArrayViewMut1<Point>,
    ) {
        let irv_ts = *self.option().irv_ts();
        let irv_th = *self.option().irv_th();
        let disp_max = *self.option().max_disparity() as f32;
        for _ in 0..5 {
            for k in 0..2 {
                let process_pixels = if k == 0 {
                    mismatches.view_mut()
                } else {
                    occlusions.view_mut()
                };
                Zip::indexed(process_pixels).for_each(|_pos, p| {
                    let pos_y = *p.y();
                    let pos_x = *p.x();
                    //如果像素为位置为(0,0)则不处理
                    if pos_y == 0 && pos_x == 0 {
                        return;
                    }
                    let arm = self.cross_arms.get((pos_y, pos_x)).unwrap();
                    //获取十字交叉臂区域
                    let (left, right) = arm.horizontal_arm_range();
                    let (top, bottom) = arm.vertical_arm_range();
                    let disp_area = self
                        .disparity_left
                        .slice_mut(s![*top..*bottom, *left..*right]);
                    //构建直方图
                    let mut hist = ndhistogram!(
                        Uniform::new(disp_max as usize, 0f32, disp_max);u32
                    );
                    disp_area.for_each(|v| {
                        if v.is_nan() {
                            return;
                        }
                        hist.fill(&v.round());
                    });
                    // 获得直方图中最大值
                    let max_val = *hist.values().max().unwrap() as f32;
                    if max_val > 0f32 {
                        //将直方图转换为一维数组
                        let a_hist = Array1::from_iter(hist.values().map(|v| *v).into_iter());
                        let count = a_hist.sum() as f32;
                        let best_disp = a_hist.argmax().unwrap() as f32;
                        if count > irv_ts && max_val / count > irv_th {
                            // 以最佳视差值填充当前元素
                            let d = self.disparity_left.get_mut((pos_y, pos_x)).unwrap();
                            *d = best_disp - 1f32;
                            //标记当前像素已填充, 下一次迭代时不再处理
                            p.set_x(0);
                            p.set_y(0);
                        }
                    }
                });
            }
        }
    }
    /// 左右一致性检查  
    fn lr_check(&mut self) -> (Vec<Point>, Vec<Point>) {
        let threshold = *self.option().lrcheck_thres();
        let width = *self.width();
        let mismatches = Arc::new(Mutex::new(Vec::<Point>::new()));
        let occlusions = Arc::new(Mutex::new(Vec::<Point>::new()));
        let mut tmp_disparity_left = self.disparity_left.clone();
        Zip::indexed(&mut tmp_disparity_left).par_for_each(|(y, x), disp_val| {
            if disp_val.is_nan() {
                //trace!("mismatch point({}, {}), val:{}", y, x, disp_val);
                //mismatches.push(Point::new(x,y));
                mismatches.lock().unwrap().push(Point::new(x, y));
                return;
            }
            // 根据视差值找到右影像上对应的同名像素
            let r_col = (x as f32 - *disp_val).round();
            if r_col >= 0f32 && r_col < width as f32 {
                // 右影像上同名像素的视差值
                let disp_val_r = self.disparity_right.get((y, r_col as usize)).unwrap();
                // 判断两个视差值是否一致（差值在阈值内）
                if (*disp_val - disp_val_r).abs() > threshold {
                    // 区分遮挡区和误匹配区
                    // 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl
                    // if(disp_rl > disp)
                    //      pixel in occlusions
                    // else
                    //      pixel in mismatches
                    let rl_col = (r_col + disp_val_r).round();
                    if rl_col >= 0f32 && rl_col < width as f32 {
                        let disp_val_l = self.disparity_left.get((y, rl_col as usize)).unwrap();
                        if disp_val_l > disp_val {
                            //trace!("occlusions point({}, {}), val:{}", y, x, disp_val_l);
                            occlusions.lock().unwrap().push(Point::new(x, y));
                        } else {
                            //trace!("mismatch point({}, {}), val:{}", y, x, disp_val_l);
                            mismatches.lock().unwrap().push(Point::new(x, y));
                        }
                    }
                    *disp_val = f32::NAN;
                }
            } else {
                //trace!("mismatch point({}, {}), val:{}", y, x, disp_val);
                *disp_val = f32::NAN;
                mismatches.lock().unwrap().push(Point::new(x, y));
            }
        });
        let ret_m = mismatches.lock().unwrap().clone();
        let ret_o = occlusions.lock().unwrap().clone();
        //self.disparity_left = tmp_disparity_left;
        (ret_m, ret_o)
    }
    /// 获取立体匹配视差范围
    pub fn get_disparity_range(&self) -> i32 {
        self.option.max_disparity() - self.option.min_disparity()
    }

    pub fn build(&self) -> Self {
        self.clone()
    }
}
