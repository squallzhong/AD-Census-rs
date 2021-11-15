use getset::{CopyGetters, Getters, MutGetters, Setters};

/// ADCensus参数设置
#[derive(Getters, Setters, MutGetters, CopyGetters, Copy, Clone, Debug)]
pub struct ADCensusOption {
    /// 最小视差
    #[getset(get = "pub", set = "pub")]
    min_disparity: i32,
    /// 最大视差
    #[getset(get = "pub", set = "pub")]
    max_disparity: i32,
    /// 控制AD代价值的参数
    #[getset(get = "pub", set = "pub")]
    lambda_ad: i32,
    /// 控制Census代价值的参数
    #[getset(get = "pub", set = "pub")]
    lambda_census: i32,
    /// 十字交叉窗口的空间域参数：L1
    #[getset(get = "pub", set = "pub")]
    cross_l1: i32,
    /// 十字交叉窗口的空间域参数：L2
    #[getset(get = "pub", set = "pub")]
    cross_l2: i32,
    /// 十字交叉窗口的颜色域参数：t1
    #[getset(get = "pub", set = "pub")]
    cross_t1: i32,
    /// 十字交叉窗口的颜色域参数：t2
    #[getset(get = "pub", set = "pub")]
    cross_t2: i32,
    /// 扫描线优化参数p1
    #[getset(get = "pub", set = "pub")]
    so_p1: f32,
    /// 扫描线优化参数p2
    #[getset(get = "pub", set = "pub")]
    so_p2: f32,
    /// 扫描线优化参数tso
    #[getset(get = "pub", set = "pub")]
    so_tso: i32,
    /// Iterative Region Voting法参数ts
    #[getset(get = "pub", set = "pub")]
    irv_ts: f32,
    /// Iterative Region Voting法参数th
    #[getset(get = "pub", set = "pub")]
    irv_th: f32,
    /// 左右一致性约束阈值
    #[getset(get = "pub", set = "pub")]
    lrcheck_thres: f32,
    /// 是否检查左右一致性
    #[getset(get = "pub", set = "pub")]
    do_lr_check: bool,
    /// 是否做视差填充
    #[getset(get = "pub", set = "pub")]
    do_filling: bool,
    /// 是否做非连续区调整
    #[getset(get = "pub", set = "pub")]
    do_discontinuity_adjustment: bool
}

impl Default for ADCensusOption {
    fn default() -> Self {
        Self {
            min_disparity: 1,
            max_disparity: 64,
            lambda_ad: 10,
            lambda_census: 30,
            cross_l1: 34,
            cross_l2: 17,
            cross_t1: 20,
            cross_t2: 6,
            so_p1: 1.0,
            so_p2: 3.0,
            so_tso: 15,
            irv_ts: 20.0,
            irv_th: 0.4,
            lrcheck_thres: 1.0,
            do_lr_check: true,
            do_filling: true,
            do_discontinuity_adjustment: true
        }
    }
}

impl ADCensusOption {
    pub fn new(min_disparity: i32, max_disparity: i32) -> Self {
        let mut r = ADCensusOption::default();
        r.min_disparity = if min_disparity < 0 {0} else {min_disparity};
        r.max_disparity = max_disparity;
        r
    }
    ///
    pub fn build(&self) -> Self{
        *self
    }
}