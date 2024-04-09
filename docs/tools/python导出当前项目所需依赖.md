# 导出python当前项目所需依赖清单requirements.txt

因使用 pip freeze > requirements.txt 会导出大量与该项目无关的依赖。

如果我们只需导出当前项目所需的依赖包，还可以采用另外一种方式，使用工具：pipreqs

## 一、安装pipreqs

```shell
pip install pipreqs
```

## 二、导出依赖

```shell
pipreqs ./ --encoding=utf8
```