#### 添加用户

```sh
useradd -m username -p password -d /data/home/xxx
```

username、password处分别填上用户名、密码
-m 表示添加用户的时候顺便建立该用户权限的文件夹
-d 表示该文件夹的目录

删除用户及其文件夹

```sh
userdel -r yyl
```

修改用户密码

```bash
passwd username
```



加目录权限





linux查看用户组

要查看Linux系统中的用户组，可以使用以下命令:

```
cat /etc/group
```

该命令将显示系统中所有用户组的列表，每个用户组的信息都以一行显示。每行包含以下字段:

```
group_name:password:GID:user_list
```

- `group_name` 是用户组的名称。
- `password` 是用户组的密码字段。通常在Linux中，用户组的密码字段为空或包含一个占位符。
- `GID` 是用户组的唯一标识符（Group ID）。
- `user_list` 是属于该用户组的用户列表，多个用户名之间用逗号分隔。

请注意，查看 `/etc/group` 文件需要 root 或具有适当权限的用户才能执行此操作。



添加用户组

将新用户添加到目标目录所属的用户组中

```sh
sudo usermod -aG groupname user1
```



要将一个磁盘目录 `/hdd` 添加到用户组 `group1`，可以使用 `chgrp` 命令来更改目录的所属用户组。然后，使用 `chmod` 命令来设置用户组的权限。

以下是添加用户组权限的基本语法：

```
sudo chgrp group1 /hdd
sudo chmod g+rwx /hdd
```

解释一下上述命令中的参数：

- `sudo`：以管理员权限运行命令。
- `chgrp`：用于更改文件或目录的所属用户组。
- `group1`：目标用户组的名称。
- `/hdd`：要更改所属用户组的目录路径。

然后，我们使用 `chmod` 命令设置用户组的权限：

- `g+rwx`：表示为用户组添加读取、写入和执行权限。

运行以上命令后，目录 `/hdd` 将被设置为用户组 `group1` 所拥有，并具有读取、写入和执行权限。

请注意，执行这些操作需要以管理员权限运行命令（使用 `sudo`），否则可能会因权限不足而无法执行操作。