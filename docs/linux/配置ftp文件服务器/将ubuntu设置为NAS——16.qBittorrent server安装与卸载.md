# 将ubuntu设置为NAS——16.qBittorrent server安装与卸载

2021-11-07 11:18:11 8点赞 47收藏 11评论

**创作立场声明：**本人小白，只是想通过本平台记录一些折腾以Ubuntu系统为主的[NAS](https://www.smzdm.com/ju/sp3qr1k/)过程，同时方便自己查阅，还能赚点金币，于是就有了这个小系列。

### 前言

前面安装了Transmission-daemon，详见将《[ubuntu设置为NAS——1. Transmission-daemon安装](https://post.smzdm.com/p/aqnlg7d2/)》，在用过了一段时间后，个人感觉Transmission下载速度并不算稳定，最麻烦的是不正常重启[电脑](https://www.smzdm.com/ju/sp4x11p/)后，有可能需要重新校验种子，校验时间比较久，而且有时候会莫名其妙出现警告或红种，怎么排查也解决不了，于是就想找找是否还有更合适的[服务器](https://www.smzdm.com/fenlei/fuwuqi/)下载BT软件。然后就发现了qBittorrent server，于是就又开始折腾了（又可以水了）。但经过一段时间使用，我还是遇到了坑，由于种子数量增加，qBittorrent[内存](https://www.smzdm.com/fenlei/neicun/)占用问题越来越严重，导致我服务器运行不稳定，最后还是被我卸载。

### 1、正常安装

1.1、命令行安装

在Ubuntu下安装可以直接选择命令行方式安装

> sudo apt install qbittorrent-nox

[![安装qbittorrent-nox](https://qnam.smzdm.com/202111/07/6186b1296b2558933.png_e1080.jpg)](https://post.smzdm.com/p/avw0zpw9/pic_2/)安装qbittorrent-nox

默认端口是8080，我修改为8181后，按照下面命令直接启动webui界面

> qbittorrent-nox --webui-port=8181

[![法律声明，确定按y](https://qnam.smzdm.com/202111/07/6186b1295c83e4188.png_e1080.jpg)](https://post.smzdm.com/p/avw0zpw9/pic_3/)法律声明，确定按y

法律声明，确定按y

[![web-ui默认账号密码](https://qnam.smzdm.com/202111/07/6186b1295e2588929.png_e1080.jpg)](https://post.smzdm.com/p/avw0zpw9/pic_4/)web-ui默认账号密码

记住默认账号：**admin**，默认密码：**adminadmin**，和端口号，在PC浏览器中输入**服务器地址+端口号**登录

[![web登录界面](https://qnam.smzdm.com/202111/07/6186b129de7ec1644.png_e1080.jpg)](https://post.smzdm.com/p/avw0zpw9/pic_5/)web登录界面

[![正常使用web界面](https://qnam.smzdm.com/202111/07/6186b12a203632356.png_e1080.jpg)](https://post.smzdm.com/p/avw0zpw9/pic_6/)正常使用web界面

1.2、版本更新

突然发现不是最新版（后来发现这一版是各大PT站点推荐的版本，也算是最稳定的版本）

[![4.1.7版本](https://qnam.smzdm.com/202111/07/6186b12a1c6738446.png_e1080.jpg)](https://post.smzdm.com/p/avw0zpw9/pic_7/)4.1.7版本

应该是Ubuntu中默认apt里的源不是最新

按照官方教程，添加更新源，再进行软件升级就可以了，截图是4.3.0.1版，但最新已经是4.3.9

> sudo a[dd](https://pinpai.smzdm.com/102135/)-apt-repository ppa:qbittorrent-team/qbittorrent-stable
> sudo apt update
> sudo apt upgrade

[![更新后4.3.0.1](https://qnam.smzdm.com/202111/07/6186b12a6b7f5344.png_e1080.jpg)](https://post.smzdm.com/p/avw0zpw9/pic_8/)更新后4.3.0.1

### 2、设置开机启动

2.1 这里通过创建自定义开机服务实现，先创建系统服务

> sudo nano /etc/systemd/system/qbittorrent-nox.service

粘贴以下内容，保存ctr+o，ctr+x退出

> [Unit]
> Description=qBittorrent-nox
> After=network.target
>
> [Service]
> User=root
> Type=simple
> RemainAfterExit=yes
> ExecStart=/usr/bin/qbittorrent-nox --webui-port=8181 -d
>
> [Install]
> WantedBy=multi-user.target

网上很多教程的qbittorrent-nox.service 的配置有问题，设置**Type=forking** 可能导致用了一段时间后 shell 会卡住，无法正常开机启动，建议在ubuntu server 20 上面，将Type=forking 的配置，改成 **Type=simple** 。

注意内容我添加了**自定义端口号8181**，如果不需要指定端口号，将配置中的ExecStart=/usr/bin/qbittorrent-nox --webui-port=8181 -d，可以设置为ExecStart=/usr/bin/qbittorrent-nox -d，**默认端口号为8080**.

2.2 启动qbittorrent-nox并创建服务配置

> sudo systemctl daemon-reload
> \# 重新加载systemd守护程序
> systemctl enable qbittorrent-nox
> \# 使qbittorrent-nox.service生效
> systemctl start qbittorrent-nox
> \# 启动qbittorrent-nox

2.3 qbittorrent-nox控制命令

> systemctl start qbittorrent-nox
> \# 启动qbittorrent-nox
> systemctl restart qbittorrent-nox
> \# 重启qbittorrent-nox
> systemctl stop qbittorrent-nox
> \# 停止qbittorrent-nox
> systemctl status qbittorrent-nox
> \# 查看qbittorrent-nox状态

### 3、卸载qbittorrent-nox

如果你觉得qbittorrent不能满足自己的需求，或者没有达到预期，想要删除bittorrent客户端，可以使用系统包管理器或运行命令：

> sudo apt remove qbittorrent-nox

如果需要卸载qbittorrent和它的各种依赖，可以使用如下命令

> sudo apt -y autoremove qbittorrent-nox

如果需要卸载qbittorrent和它设置和数据文件，可以使用如下命令

> sudo apt-get -y purge qbittorrent-nox

如果需要卸载qbittorrent、它设置和数据文件以及各种依赖，可以使用如下命令

> sudo apt-get -y autoremove --purge qbittorrent-nox

### 总结

qbittorrent-nox的安装其实很简单，web-ui使用也很顺手，可能刚开始使用不一定速度很快，需要添加 [Tracker](https://trackerslist.com/#/zh?id=qbittorrent) 列表，速度就能起来了，当然如果只是用来PT下载和抢上传一般也是足够了。

在使用了一段时间后，我发现了qbittorrent很**严重**的问题，随着我下得种子越来越多，我的服务器变得很不稳定，经常死机，通过排查发现是qbittorrent得问题，它**占用了太多得内存**，直到Ubuntu系统死机。一开始我使用8G内存，心想可能我服务器上服务太多，内存不足，就加了一条8G内存，结果还是全被qbittorrent占掉了。无奈最后还是放弃了qbittorrent，在找到解决方案前，暂时是不打算用了。**如果各位网友有好得解决方案，也请不吝赐教**。





version: "3"
services:
 nas-tools:
 image: nastool/nas-tools:latest
	 ports:
    - 3000:3000 # 默认的webui控制端口
      volumes:
		- /home/media/nas-tools/config:/config # 冒号左边请修改为你想保存配置的路径
		#- /你的媒体目录:/你想设置的容器内能见到的目录 # 媒体目录，多个目录需要分别映射进来，需要满足配置文件说明中的要求
		- /home/media/movies:/data/movies
		- /home/media/disk:/data/disk
		 environment:
		- PUID=0 # 想切换为哪个用户来运行程序，该用户的uid
		- PGID=0 # 想切换为哪个用户来运行程序，该用户的gid
		- UMASK=000 # 掩码权限，默认000，可以考虑设置为022
		- NASTOOL_AUTO_UPDATE=true # 如需在启动容器时自动升级程程序请设置为true
		- REPO_URL=https://ghproxy.com/https://github.com/NAStool/nas-tools.git # 当你访问github网络很差时，可以考虑解释本行注释
		 restart: always
		  network_mode: bridge
		  hostname: nas-tools
		  container_name: nas-tools