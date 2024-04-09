### YUV数据-操作指定像素点数据,保存yuv文件

YUV数据-操作指定像素点数据,保存yuv文件

1.YUV格式有两大类：planar和packed。

对于planar的YUV格式，先连续存储所有像素点的Y，紧接着存储所有像素点的U，随后是所有像素点的V。

对于packed的YUV格式，每个像素点的Y,U,V是连续交*存储的。

本次讲解的是常用的YUV420（I420）格式，该格式采用planar格式。

该格式对应的数据采集，可以参考下图：

![img](https://app.yinxiang.com/FileSharing.action?hash=1/ead49e089fd0931372fe4d4b23cbde93-31516)

![img](https://app.yinxiang.com/FileSharing.action?hash=1/64a1625cdfce17bd615b0c7895446601-10271)

Y'00、Y'01、Y'10、Y'11共用Cr00、Cb00，也就是两行两列Y数据对应一行一列UV数据。

对应一帧宽为W，高为H的YUV数据。数据的大小为W*H*3/2个字节，因为像素点有W*H个，对应的Y数据有W*H个字节，4个Y对应一个U和一个V，所以U/V分别各有W*H/4个字节。

YUV数据为unsigned char类型而不是char类型。并且数据是按照行优先存储的，也就是先存放了第一行的Y，再存放第二行Y，将Y存储完后再存储U/V，U/V内部也是按照行优先存储。

以下为常见读取一帧、读取指定像素点、YUV赋值、保存YUV文件

1.打开YUV文件，读取一帧YUV数据。

2.根据坐标值求对应YUV

坐标点位x,y -->

如果直接把读取UV的公式写成y*w/4  则U的会根据y的每一行增加而增加，现在是只有当y为偶数时候，才会增加。切记是两行两列Y对应一行一列UV。

int temp = y/2;  

Y:  y*w +x

U:  w*h + temp*w/2 + x /  2

V:  w*h + w*h /4  + temp*w/2 + x /  2

转为C++代码则是：

int temp = y/2;

unsigned char Y = FrameData[w*y+x];

unsigned char U = FrameData[w*h+w/2*temp+x/2];

unsigned char V = FrameData[w*h*5/4+w/2*temp+x/2];

3.如何给YUV数据赋值

//蓝色0x335681  像素点的Y为80,0x50  U值为155，0x9b,V值为107，0x6b

将FrameData指针指向此帧frame。

unsigned char Save_Y =80;

unsigned char Save_U=155;

unsigned char Save_V=107;

memset(FrameData,Save_Y,w*h);

memset(FrameData+(w*h),Save_U,w*h/4);

memset(FrameData+(w*h*5/4),Save_V,w*h/4);

tip:请注意memset给char或unsigned类型赋值时候，是可以赋不超过类型容纳的值，[0,256],U/V的最大值为256。但是对于int类型，memset只能赋0。关于memset的用法，可以自行搜索更多资料。

4.保存YUV文件

​    （window操作）

​    将FrameData指向解码后或者自己手动生成的一帧frame。

​    FILE * fn1 = fopen("f:\\0.yuv","ab+");

​    fwrite(FrameData, 1,w*h*3/2,fn1);

​    fclose(fn1);