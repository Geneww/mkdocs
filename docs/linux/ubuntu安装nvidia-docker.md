### **安装Nvidia-Docker**

Docker也是虚拟化环境的神器，前面说的conda虽然可以提供python的虚拟环境并方便地切换，但是有的时候我们的开发环境并不只是用到python，比如有的native库需要对应gcc版本的编译环境，或者进行交叉编译时哟啊安装很多工具链等等。如果这些操作都在服务器本地上进行，那时间久了就会让服务器的文件系统非常杂乱，而且还会遇到各种软件版本冲突问题。

Docker就可以很好地解决这些问题，它其实可以理解为就是一个非常轻量化的虚拟机，我们可以在宿主服务器上新建很多个这种被称为`容器`的虚拟机，然后在里面配置我们的开发环境，且这些配置好的环境是可以打包成`镜像`的，方便随时做分享和重用；不需要的时候，我们直接删除容器就好了，其资源是和我们的服务器宿主机完全隔离的。

Docker的具体使用可以自己搜索一下很多教程，这里主要介绍如何把GPU暴露给Docker的容器（因为大家都知道像是VMware这种虚拟机里面都是无法共享宿主机的GPU的），是通过`nvidia-docker`实现的。

> 以前为了配置nvidia-docker，需要安装完docker之后再安装单独的nvidia docker2，而现在只需要安装nvidia container toolkit即可，更加方便了。

1. docker安装 官网上有详细的介绍：[Install Docker Engine on Ubuntudocs.docker.com](https://link.zhihu.com/?target=https%3A//docs.docker.com/engine/install/ubuntu/) 或者运行下面的命令安装：

```shell
sudo apt-get update
sudo apt-get install docker.io
systemctl start docker
systemctl enable docker
```

可以运行这条命令检查是否安装成功：

```shell
docker version
```

2. 安装NVIDIA Container Toolkit

官网安装步骤：[NVIDIA/nvidia-docker: Build and run Docker containers leveraging NVIDIA GPUs (github.com)](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/nvidia-docker) 或者直接运行下面的命令：

```shell
 ##首先要确保已经安装了nvidia driver
 # 2. 添加源
 distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
 curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
 curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
 
 # 2. 安装并重启
 sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
 sudo systemctl restart docker
```

安装完成后可以新建一个容器测试一下：

```shell
sudo docker run -it --rm --name test_nvidia_docker --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

其中最后的参数`nvidia/cuda:11.1-base` 是Nvidia官方的镜像，需要根据工作站主机中实际安装的cuda版本进行修改，版本可以用`nvcc -V`查看。

进入容器之后可以跑一下`nvidia-smi`命令

![image-20231218175130025](/Users/gene/Library/Application Support/typora-user-images/image-20231218175130025.png)