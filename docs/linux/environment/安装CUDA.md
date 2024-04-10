### **安装CUDA**

[CUDA下载](https://developer.nvidia.com/cuda-downloads)

```shell
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

cuda安装后日志输出

```
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-11.8/

Please make sure that
 -   PATH includes /usr/local/cuda-11.8/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.8/lib64, or, add /usr/local/cuda-11.8/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.8/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 520.00 is required for CUDA 11.8 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
```

### **安装CuDNN**

进入到CUDNN的下载官网：[cuDNN Download | NVIDIA Developer](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/rdp/cudnn-download)，然点击Download开始选择下载版本，当然在下载之前还有登录，选择版本界面如下：

![img](https://pic4.zhimg.com/80/v2-b39bf69766e2bff31548cd2ccbb6e003_720w.webp)

我们选择和之前cuda版本对应的cudnn版本：

![img](https://pic1.zhimg.com/80/v2-e4c256e6cee42ba18a2c01de6c3798e4_720w.webp)

下载之后是一个压缩包，对它进行解压，命令如下：

```text
 tar -xzvf cudnn-11.0-linux-x64-v8.0.5.39.tgz
```

使用以下两条命令复制这些文件到CUDA目录下：

```text
sudo cp lib/* /usr/local/cuda-11.8/lib64/
sudo cp include/* /usr/local/cuda-11.8/include/
```

拷贝完成之后，可以使用以下命令查看CUDNN的版本信息：

```text
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

可以看到版本信息如下，为`8.0.5`：

![img](https://pic2.zhimg.com/80/v2-61030edf86acd862ac404470b05988c5_720w.webp)