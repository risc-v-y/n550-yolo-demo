# Host 视频离线依赖

适用：Linux x86_64、CPython 3.8、glibc 2.17+。用于本项目的视频解码、预处理、绘框、窗口显示和MP4编码。

原始wheel未经修改，文件SHA256已与官方PyPI发布记录核对：

- [NumPy 1.24.4](https://pypi.org/project/numpy/1.24.4/)：CPython 3.8 manylinux2014 x86_64。
- [OpenCV 4.11.0.86](https://pypi.org/project/opencv-python/4.11.0.86/)：CPython 3.7+ abi3 manylinux2014 x86_64，包含GUI支持。
- [pip 24.1.2](https://pypi.org/project/pip/24.1.2/)：只用于从wheel启动离线安装，host不必预装pip。

在仓库根目录执行：

```bash
# 校验官方安装包，安装到项目独立目录，并检查host的视频前后处理及录像
bash tools/setup_host_video.sh

# 在已有图形会话中检查显示、加载固件并处理前三帧
bash tools/host.sh video
```

安装过程关闭网络软件源，不需要改变host时间或安装系统Python包。安装输出位于`yolo26_pc/.venv-host-video/lib/`，视频入口自动加载。操作系统动态库和可用的X11会话由host提供；安装后的真实检查会报告缺失项。
