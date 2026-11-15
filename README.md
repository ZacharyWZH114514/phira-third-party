# Phira 第三方修改版

在 [TeamFlos/phira](https://github.com/TeamFlos/phira) v0.8.3 基础上做的自定义构建，主要改动：

## 音频
- 新增 **纯 Rust JNI `android.media.AudioTrack` 播放后端**（`sasa/src/backend/audiotrack.rs`），支持 `ENCODING_PCM_FLOAT`、阻塞写入、独立渲染线程、`USAGE_GAME`
- 设置 → 音频 增加 **音频后端切换**（AAudio / AudioTrack）
- 音频缓冲区大小可调（128 / 256 / 512 …），AAudio 也生效
- mixer 采样率每块钉死为设备实际输出率，避免变调 / 变速
- 主页 BGM 使用独立的大缓冲管理器，不再受游戏后端影响

## 外观
- **自定义人物形象**：设置 → 自定义 → 选图，按图片宽高比取景、底边留 30px、长按恢复默认
- **自定义背景**：替换主页背景图（重启生效）
- **自定义主页音乐**：支持 mp3 / aac / ogg / wav / flac / m4a（重启生效）
- 备案号移到右下角，并按背景亮度自动调整文字深浅
- 设置页新增「自定义」分区

## 其它
- **离线成绩记录**：一局结束后本地落盘，谱面详情页可显示
- **侧边返回手势**：从屏幕左边缘向右滑动 = 返回上一级
- 关于页版本号显示构建者标识

## 说明
- 本仓库只包含**修改过的源文件**，把它们覆盖到 Phira v0.8.3 源码树对应路径即可（另需把 `sasa` vendor 进 workspace）。
- 官方 Android 壳为闭源，本仓库不含、也不再分发其二进制。
- Phira 采用 GPL-3.0，本仓库的修改同样以 GPL-3.0 发布。

## 构建
参见 Phira 官方文档；Android 交叉编译需 NDK r27c，并用 `-Clink-arg=-Wl,-z,max-page-size=16384` 适配 16KB 页。
