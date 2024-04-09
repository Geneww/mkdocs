实验室服务器装了双系统，一个是Ubuntu，一个是Windows10。Win10是作为日常使用的系统，用来搭建FTP服务器。但最近多用来跑训练，所以FTP服务器就无法使用，造成很多不便。因此希望切换到Ubuntu下也要能作为FTP服务器。下面开始折腾。

首先假设我在用户A下面，先安装上vsftpd，

sudo apt install vsftpd
配置vsftpd.conf,

/etc/vsftpd/vsftpd.conf：配置文件
/usr/sbin/vsftpd：Vsftpd的主程序
/etc/rc.d/init.d/vsftpd ：启动脚本

anonymous_enable=NO # 禁止匿名登陆
local_enable=YES # 允许本地登录
write_enable=YES
utf8_filesystem=YES
allow_writable_chroot=YES
local_root=/home/A # A登陆上的目录，注意要在home目录下
这样配置之后重启vsftpd服务，

sudo systemctl restart vsftpd.service
就可以使用A和A的密码登陆上FTP服务器了，路径就是刚刚设定的local_path。但是这样设定的毛病是登陆上之后登录用户取得了整个/目录下的权限，非常不安全。所以我们接下来再往vsftpd.conf里面添加

chroot_local_user=NO
chroot_list_enable=YES
chroot_list_file=/etc/vsftpd.chroot_list # 将登陆用户名添加到这里限定访问目录
# local_root=/home/A # 注释掉该句
user_config_dir=/etc/vsftpd_user_conf # 添加该句以设置不同用户的目录
并创建vsftpd.chroot_list文件，在其中添加登陆用户名A，

# /etc/vsftpd.chroot_list
A
创建vsftpd_user_conf文件夹，里面创建A文件，添加

# /etc/vsftpd_user_conf/A
local_root=/home/A
这样可以将A的访问目录限制在local_path当中。如果需要多个账号登陆及指定，例如另一个B账号，指定在目录/home/B（已创建好的home下的任意子文件夹），则可以使用

sudo useradd -d /home/B -M -s /usr/sbin/nologin B
sudo passwd B
创建一个只允许登陆FTP的根目录在/home/B中的账号B，注意此时要将B也添加到vsftpd.chroot_list文件当中，同时创建/vsftpd_user_conf/B，添加

# /etc/vsftpd_user_conf/B
local_root=/home/B
Windows创建的NTFS分区下的目录在挂载到/media或者/mnt下时，作为FTP目录总会报

错误。网上很多教程是说selinux的错误，但是环境中selinux已经关闭。因此只能选择将其挂载到/home目录下，因此在/home下可以创建一个ftp文件夹，其下挂载Windows下用来存储FTP服务器数据的硬盘。具体操作类似于

# /etc/fstab
/dev/sda2 /home/ftp/ ntfs defaults 0 0
即可。随后再指定用户的目录

sudo usermod -d /home/ftp/B B

# /etc/vsftpd.conf
local_root = /home/ftp/A
这样即可达到Windows与Ubuntu公用一块硬盘作为FTP服务器的效果。