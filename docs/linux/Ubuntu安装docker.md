# [Ubuntu18.04安装docker](https://www.cnblogs.com/wt7018/p/11880666.html)

参考

https://www.runoob.com/docker/ubuntu-docker-install.html

1.卸载

```
sudo apt-get remove docker docker-engine docker.io containerd runc
```

2.安装Docker

```
sudo apt-get update
# 安装依赖包
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
# 添加 Docker 的官方 GPG 密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# 验证您现在是否拥有带有指纹的密钥
sudo apt-key fingerprint 0EBFCD88
# 设置稳定版仓库
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

![img](https://img2018.cnblogs.com/i-beta/1661179/201911/1661179-20191118102714835-1684256926.png)

3.安装 Docker Engine-Community

```
# 更新
$ sudo apt-get update
# 安装最新的Docker-ce 
sudo apt-get install docker-ce
# 启动
sudo systemctl enable docker
sudo systemctl start docker
```

4.测试

```
sudo docker run hello-world
```

![img](https://img2018.cnblogs.com/i-beta/1661179/201911/1661179-20191118102729705-1522389123.png)

 

在用户权限下docker 命令需要 sudo 否则出现以下问题

![img](https://img-blog.csdnimg.cn/20200227175406444.png)

通过将用户添加到docker用户组可以将sudo去掉，命令如下

sudo groupadd docker #添加docker用户组

sudo gpasswd -a $USER docker #将登陆用户加入到docker用户组中

newgrp docker #更新用户组