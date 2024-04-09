# FRP搭建

frp 是一个高性能的反向代理应用，可以帮助您轻松地进行内网穿透

frp 采用 C/S 模式，将服务端部署在具有公网 IP 机器上，客户端部署在内网或防火墙内的机器上，通过访问暴露在服务器上的端口，反向代理到处于内网的服务。 

## 一、下载frp

[frp下载地址](https://github.com/fatedier/frp/releases)

```shell
wget https://github.com/fatedier/frp/releases/download/v0.57.0/frp_0.57.0_linux_amd64.tar.gz
```

## 二、解压frp至服务端

```shell
tar -zxvf frp_0.57.0_linux_amd64.tar.gz && mv frp_0.57.0_linux_amd64 frp && cd frp
```

## 三、配置frp服务端

修改frps.ini配置文件，内容如下：

```shell
[common]
bind_port = 7000 # frp服务的端口号，可以自己定
dashboard_port = 7500 # frp的web界面的端口号
dashboard_user = user # web界面的登陆账户，自己修改
dashboard_pwd = pass # web界面的登陆密码，自己修改
authentication_method = token
token = xxxxx # frp客户端连接时的密码，自己修改
# console or real logFile path like ./frps.log
log_file=/usr/local/bin/frp/logs/frps.log
# trace, debug, info, warn, error
log_level = info
log_max_days = 3
# disable log colors when log_file is console, default is false
disable_log_color = false
```

在浏览器输入 `[云服务器的公网ip]:7500` 即可访问到 frp的web管理界面。

> 注意，可能需要去云服务器控制台配置安全组规则 开放以上涉及到的端口，否则无法访问。

## 四、添加frp服务端为系统服务开机自启

添加开机自动启动的脚本，在/etc/systemd/system/目录下新建一个frps.service文件内容如下：

完整路径`/etc/systemd/system/frps.service`

```shell
[Fusion]
Description=Frp Server Daemon
After=syslog.target network.target
Wants=network.target

[Service]
Type=simple
ExecStart=/root/frp/frps -c /root/frp/frps.ini # 修改为你的frp实际安装目录
ExecStop=/usr/bin/killall frps
#启动失败1分钟后再次启动
RestartSec=1min
KillMode=control-group
#重启控制：总是重启
Restart=always

[Install]
WantedBy=multi-user.target
```

然后执行以下命令启用脚本：

```shell
sudo systemctl enable frps.service
sudo systemctl start frps.service
```

通过下面的命令查看服务状态，如果是running的话就说明可以了：

```shell
sudo systemctl status frps.service
```

<img src="https://image.iokko.cn/file/6045d514d8b8598b15ca1.png" alt="Image" style="float: left;">

## 五、解压frp至客户端

```shell
tar -zxvf frp_0.57.0_linux_amd64.tar.gz && mv frp_0.57.0_linux_amd64 frp && cd frp
```

## 六、配置frp客户端

修改frpc.ini配置文件，内容如下：

```shell
[common]
server_addr=xx.xx.xx.xx # 你的云服务器的公网ip
server_port=7000
token=xxx
log_file=/data/nfsroot/frp/frpc.log
# ssh
[home_server_ssh]
type = tcp
local_ip = 127.0.0.1
local_port = 22
remote_port = 20022
# rdp
[home_server_rdp]
type = tcp
local_ip = 127.0.0.1
local_port = 3389
remote_port = 23389

#这步可选开通web端
[home_server_web]
type=http
local_ip=127.0.0.1
local_port=80
subdomain=xx
```

通过上面的脚本就可以把对于云服务器特定端口的访问给重定向到本地服务器的某个端口了，简单地讲就是：假如我用SSH客户端访问 `[云服务器ip]:20022`，就可以经过反向代理直接访问到`[本地的训练服务器ip]:22`；同理需要连接远程桌面的话，只需要访问`[云服务器ip]:23389`就可以了。

## 七、添加frp客户端为系统服务开机自启

按照上四描述添加开机自动启动的脚本，在/etc/systemd/system/目录下新建一个frpc.service文件内容如下：

完整路径`/etc/systemd/system/frpc.service`

```shell
[Fusion]
Description=Frpc Server Daemon
After=syslog.target network.target
Wants=network.target
 
[Service]
Type=simple
ExecStart=/usr/local/bin/frp/frpc -c /usr/local/bin/frp/frpc.ini # 修改为你的frp实际安装目录
ExecStop=/usr/bin/killall frpc
#启动失败1分钟后再次启动
RestartSec=1min
KillMode=control-group
#重启控制：总是重启
Restart=always
 
[Install]
WantedBy=multi-user.target
```

然后执行以下命令启用脚本：

```shell
sudo systemctl enable frpc.service
sudo systemctl start frpc.service
```

通过下面的命令查看服务状态，如果是running的话就说明可以了：

```shell
sudo systemctl status frpc.service
```

> 这里顺便提一下，按照习惯一般把上面的frp软件解压防止在`/usr/local/bin`目录下。Linux 的软件安装目录是也是有讲究的，理解这一点，在对系统管理是有益的

- `/usr`：系统级的目录，可以理解为`C:/Windows/`
- `/usr/lib`：可以理解为`C:/Windows/System32`
- `/usr/local`：用户级的程序目录，可以理解为`C:/Progrem Files/`，用户自己编译的软件默认会安装到这个目录下
- `/opt`：用户级的程序目录，可以理解为`D:/Software`，opt有可选的意思，这里可以用于放置第三方大型软件（或游戏），当你不需要时，直接`rm -rf`掉即可。在硬盘容量不够时，也可将`/opt`单独挂载到其他磁盘上使用

> 源码放哪里？

- `/usr/src`：系统级的源码目录
- `/usr/local/src`：用户级的源码目录。