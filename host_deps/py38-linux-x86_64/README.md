# Host 视频离线依赖

适用：Linux x86_64、CPython 3.8、glibc 2.17+。用于本项目的视频解码、预处理、绘框、窗口显示和MP4编码。

原始wheel未经修改，文件SHA256已与官方PyPI发布记录核对：

- [NumPy 1.24.4](https://pypi.org/project/numpy/1.24.4/)：CPython 3.8 manylinux2014 x86_64。
- [OpenCV 4.11.0.86](https://pypi.org/project/opencv-python/4.11.0.86/)：CPython 3.7+ abi3 manylinux2014 x86_64，包含GUI支持。
- [pip 24.1.2](https://pypi.org/project/pip/24.1.2/)：只用于从wheel启动离线安装，host不必预装pip。

安装及视频运行命令只在[操作手册](../../docs/OPERATIONS.md#1-准备与更新)维护。本页只记录依赖来源、平台条件及系统兼容处理。

安装过程关闭网络软件源，不需要改变host时间或安装系统Python包。安装输出位于`yolo26_pc/.venv-host-video/lib/`，视频入口自动加载。操作系统动态库和可用的X11会话由host提供；安装后的真实检查会报告缺失项。

安装脚本在本次pip进程中启用其受支持的`sysconfig`路径选择，避免Python3.8系统`distutils`组件不匹配时启动失败。该选项不写入系统配置，原始wheel及其哈希保持不变。遇到`distutils/sysconfig`导入错误时，更新仓库后重新执行操作手册的安装入口即可；无需修补系统Python或重新运行板端。
