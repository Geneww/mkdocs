# 完美解决ubuntu20中tracker占用过多cpu，硬盘的bug

最近运行程序，一段时间后被提示硬盘上没有可用的空间了，用磁盘查看器一看原来是 .cache下的tracker占用了我70GB的硬盘！！简直丧心病狂，而且和一个贪婪的野兽一样，还在疯狂的增长！！

看了很多帖子都不能很好的解决，最后终于找到了方法，特地搬运过来：

track是linux中的文件索引工具，ubuntu18之前是默认不安装的，所以在升级到20后会默认安装，它是和桌面程序绑定的，甚至还有很多依赖项，导致无法删除，一旦删除很多依赖项都不能运行，禁用也很难禁用的掉，而且禁用了还会导致其他应用程序启动失败，这里介绍一种更好的方法。

1.首先打开终端输入下面的命令，目的是屏蔽tracker systemd服务，完全禁用当前的服务

```bash
systemctl --user mask tracker-store.service tracker-miner-fs.service tracker-miner-rss.service tracker-extract.service tracker-miner-apps.service tracker-writeback.service
```

2.然后重启跟踪器：

```text
tracker reset --hard
```

3.重启系统reboot

重启后检测：

```text
tracker status
```

显示“无法建立到 Tracker 的连接: Failed to load SPARQL backend: GDBus.Error:org.freedesktop.systemd1.UnitMasked: Unit tracker-store.service is masked.”说明已经屏蔽

还可以检测tracker的辅助程序是否被禁用：

```text
tracker daemon
```

显示下面的说明被成功屏蔽

![img](https://pic3.zhimg.com/80/v2-4be3cae3c1ec14d1ba1ccec179674a02_720w.webp)

最后进入到.cache/tracker下删掉那个最大的文件就行了

如果想撤销操作，恢复跟踪器：

```text
systemctl --user unmask tracker-store.service tracker-miner-fs.service tracker-miner-rss.service tracker-extract.service tracker-miner-apps.service tracker-writeback.service
```

其他的方法：

方法1：

```text
gsettings set org.freedesktop.Tracker.Miner.Files crawling-interval -2
```

方法2：停止后台的tracker

```text
tracker daemon -t
```