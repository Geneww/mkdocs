准备工作
Centos需安装epel-release源
#CentOS
yum install epel-release
yum -y update
Ubuntu使用Transmission官方源安装
#Ubuntu
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:transmissionbt/ppa
安装Transmission
#CentOS
yum install transmission-cli transmission-common transmission-daemon

#Ubuntu/Debian
sudo apt-get install transmission-daemon -y
修改Transmission Web配置文件
#停止Transmisson后台服务
systemctl stop transmission-daemon.service

#修改配置文件
vim /var/lib/transmission-daemon/.config/transmission-daemon/settings.json
Settings.json文件如下
{
    "alt-speed-down": 50,
    "alt-speed-enabled": false,
    "alt-speed-time-begin": 540,
    "alt-speed-time-day": 127,
    "alt-speed-time-enabled": false,
    "alt-speed-time-end": 1020,
    "alt-speed-up": 50,
    "bind-address-ipv4": "0.0.0.0",
    "bind-address-ipv6": "::",
    "blocklist-enabled": false,
    "blocklist-url": "<http://www.example.com/blocklist>",
    "cache-size-mb": 4,
    "dht-enabled": true,
    "download-dir": "/var/lib/transmission-daemon/downloads",
    "download-limit": 100,
    "download-limit-enabled": 0,
    "download-queue-enabled": true,
    "download-queue-size": 5,
    "encryption": 1,
    "idle-seeding-limit": 30,
    "idle-seeding-limit-enabled": false,
    "incomplete-dir": "/var/lib/transmission-daemon/Downloads",
    "incomplete-dir-enabled": false,
    "lpd-enabled": false,
    "max-peers-global": 200,
    "message-level": 1,
    "peer-congestion-algorithm": "",
    "peer-id-ttl-hours": 6,
    "peer-limit-global": 200,
    "peer-limit-per-torrent": 50,
    "peer-port": 51413,
    "peer-port-random-high": 65535,
    "peer-port-random-low": 49152,
    "peer-port-random-on-start": false,
    "peer-socket-tos": "default",
    "pex-enabled": true,
    "port-forwarding-enabled": false,
    "preallocation": 1,
    "prefetch-enabled": true,
    "queue-stalled-enabled": true,
    "queue-stalled-minutes": 30,
    "ratio-limit": 2,
    "ratio-limit-enabled": false,
    "rename-partial-files": true,
    "rpc-authentication-required": true,
    "rpc-bind-address": "0.0.0.0",
    "rpc-enabled": true,
    "rpc-host-whitelist": "",
    "rpc-host-whitelist-enabled": true, 
    "rpc-password": "这里填写你的登陆密码",
    "rpc-port": 9091, #外网访问端口可以自行更改
    "rpc-url": "/transmission/",
    "rpc-username": "这里填写你的登陆账号",
    "rpc-whitelist": "127.0.0.1",
    "rpc-whitelist-enabled": false, #默认为true, 如需从外网访问最好将白名单关掉或将自己的ip添加至白名单
    "scrape-paused-torrents-enabled": true,
    "script-torrent-done-enabled": false,
    "script-torrent-done-filename": "",
    "seed-queue-enabled": false,
    "seed-queue-size": 10,
    "speed-limit-down": 100,
    "speed-limit-down-enabled": false,
    "speed-limit-up": 100,
    "speed-limit-up-enabled": false,
    "start-added-torrents": true,
    "trash-original-torrent-files": false,
    "umask": 18,
    "upload-limit": 100,
    "upload-limit-enabled": 0,
    "upload-slots-per-torrent": 14,
    "utp-enabled": true
}
放行Transmission Web端口这里为9091，具体方法跳转至下文
Linux服务器脚本合集

启动Transmission后台服务
#启动transmission服务
systemctl start transmission-daemon.service

#停止transmission服务
systemctl stop transmission-daemon.service

#查询transmission运行状态
systemctl status transmission-daemon.service

#将transmission设置为开机自启动
systemctl enable transmission-daemon.service

#关闭transmission开机自启
systemctl disable transmission-daemon.service
更换Transmission Web Control美化主题（可选）
wget <https://github.com/ronggang/transmission-web-control/raw/master/release/install-tr-control-cn.sh>
bash install-tr-control-cn.sh
#然后选择1运行脚本
在浏览器通过ip:port进入Transmission的Web端页面
