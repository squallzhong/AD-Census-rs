use getset::{CopyGetters, Getters, MutGetters, Setters};
/// 最大臂长
//const MAX_ARM_LENGTH:u8 = 255u8;

/// 交叉十字臂结构, 臂长最长不能超过255
#[derive(Getters, Setters, MutGetters, CopyGetters, Copy, Clone, Debug)]
pub struct CrossArm {
    ///左臂长度
    #[getset(get = "pub", set = "pub")]
    left: u8,
    ///右臂长度
    #[getset(get = "pub", set = "pub")]
    right: u8,
    ///上臂长度
    #[getset(get = "pub", set = "pub")]
    top: u8,
    ///下臂长度
    #[getset(get = "pub", set = "pub")]
    bottom: u8,
    ///横向臂起止范围
    #[getset(get = "pub", set = "pub")]
    horizontal_arm_range: (usize, usize),
    ///纵向臂起止范围
    #[getset(get = "pub", set = "pub")]
    vertical_arm_range: (usize, usize),
    /// 交叉十字臂覆盖的区域像素的数量-先横向再纵向-P1
    #[getset(get = "pub", set = "pub")]
    pixel_count_in_area_by_horizontal: i32,
    /// 交叉十字臂覆盖的区域像素的数量-先纵向再横向-P2
    #[getset(get = "pub", set = "pub")]
    pixel_count_in_area_by_vertical: i32,
}
impl CrossArm {
    /// 获取当前像素横向臂方向像素数量(不包括自己本身)
    pub fn horizontal_pixel_count(&self) -> i32 {
        (self.left + self.right) as i32
    }
    /// 获取当前像素纵向臂方向像素数量(不包括自己本身)
    pub fn vertical_pixel_count(&self) -> i32 {
        (self.top + self.bottom) as i32
    }
    /// 获取交叉十字臂四个方向臂长
    /// 返回数据
    ///     (左臂数量, 右臂数量, 上臂数量, 下臂数量)
    pub fn arm_len(&self) -> (i32, i32, i32, i32) {
        (
            self.left as i32,
            self.right as i32,
            self.top as i32,
            self.bottom as i32,
        )
    }
    /// 交叉十字臂覆盖的区域像素的数量, P1和P2像素总和
    pub fn pixel_count_in_area(&self) -> i32 {
        self.pixel_count_in_area_by_vertical + self.pixel_count_in_area_by_horizontal
    }
}
impl Default for CrossArm {
    fn default() -> Self {
        Self {
            left: 0u8,
            right: 0u8,
            top: 0u8,
            bottom: 0u8,
            pixel_count_in_area_by_horizontal: 0i32,
            pixel_count_in_area_by_vertical: 0i32,
            vertical_arm_range: (0usize, 0usize),
            horizontal_arm_range: (0usize, 0usize),
        }
    }
}
