# 用python搭建gRPC服务

[参考地址](https://bbs.huaweicloud.com/community/usersnew/id_1568284004877096)

 【摘要】 gRPC是一个高性能、通用的开源RPC框架，其由Google主要面向移动应用开发并基于HTTP/2协议标准而设计，基于ProtoBuf序列化协议开发，且支持众多开发语言。一个gRPC服务的大体结构图为：图一表明，grpc的服务是跨语言的，但需要遵循相同的协议（proto）。相比于REST服务，gPRC 的一个很明显的优势是它使用了二进制编码，所以它比 JSON/HTTP 更快，且有清晰的接口...

gRPC是一个高性能、通用的开源RPC框架，其由Google主要面向移动应用开发并基于HTTP/2协议标准而设计，基于ProtoBuf序列化协议开发，且支持众多开发语言。一个gRPC服务的大体结构图为：

![img](https://bbs-img.huaweicloud.com/blogs/img/images_162453763267773.png)

图一表明，grpc的服务是跨语言的，但需要遵循相同的协议（proto）。相比于REST服务，gPRC 的一个很明显的优势是它使用了二进制编码，所以它比 JSON/HTTP 更快，且有清晰的接口规范以及支持流式传输，但它的实现相比rest服务要稍微要复杂一些，下面简单介绍搭建gRPC服务的步骤。

1. **安装python需要的库**

   ```shell
   pip install grpcio
   pip install grpcio-tools  
   pip install protobuf
   ```

2. **定义gRPC的接口**

   创建 gRPC 服务的第一步是在.proto 文件中定义好接口，proto是一个协议文件，客户端和服务器的通信接口正是通过proto文件协定的，可以根据不同语言生成对应语言的代码文件。这个协议文件主要就是定义好服务（service）接口，以及请求参数和相应结果的数据结构，具体的proto语法参见如下链接（https://www.jianshu.com/p/da7ed5914088），关于二维数组、字典等python中常用的数据类型，proto语法的表达见链接（https://blog.csdn.net/xiaoxiaojie521/article/details/106938519），下面是一个简单的例子。

   ```javascript
   syntax = "proto3";
   
   option cc_generic_services = true;
   
   //定义服务接口
   service GrpcService {
       rpc hello (HelloRequest) returns (HelloResponse) {}  //一个服务中可以定义多个接口，也就是多个函数功能
   }
   
   //请求的参数
   message HelloRequest {
       string data = 1;   //数字1,2是参数的位置顺序，并不是对参数赋值
       Skill skill = 2;  //支持自定义的数据格式，非常灵活
   };
   
   //返回的对象
   message HelloResponse {
       string result = 1;
       map<string, int32> map_result = 2; //支持map数据格式，类似dict
   };
   
   message Skill {
       string name = 1;
   };
   ```

3. **使用 protoc 和相应的插件编译生成对应语言的代码**

   ```javascript
   python -m grpc_tools.protoc -I ./ --python_out=./ --grpc_python_out=. ./hello.proto
   ```

   利用编译工具把proto文件转化成py文件，直接在当前文件目录下运行上述代码即可。

   - -I 指定proto所在目录
   - -m 指定通过protoc生成py文件
   - --python_out指定生成py文件的输出路径
   - hello.proto 输入的proto文件

   执行上述命令后，生成hello_pb2.py 和hello_pb2_grpc.py这两个文件。

4. **编写grpc的服务端代码**

   ```javascript
   #! /usr/bin/env python
   # coding=utf8
   
   import time
   from concurrent import futures
   
   import grpc
   
   import hello_pb2_grpc, hello_pb2
   
   _ONE_DAY_IN_SECONDS = 60 * 60 * 24
   
   
   class TestService(hello_pb2_grpc.GrpcServiceServicer):
       '''
       继承GrpcServiceServicer,实现hello方法
       '''
       def __init__(self):
           pass
   
       def hello(self, request, context):
           '''
           具体实现hello的方法，并按照pb的返回对象构造HelloResponse返回
           :param request:
           :param context:
           :return:
           '''
           result = request.data + request.skill.name + " this is gprc test service"
           list_result = {"12": 1232}
           return hello_pb2.HelloResponse(result=str(result),
                                          map_result=list_result)
   
   def run():
       '''
       模拟服务启动
       :return:
       '''
       server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
       hello_pb2_grpc.add_GrpcServiceServicer_to_server(TestService(),server)
       server.add_insecure_port('[::]:50052')
       server.start()
       print("start service...")
       try:
           while True:
               time.sleep(_ONE_DAY_IN_SECONDS)
       except KeyboardInterrupt:
           server.stop(0)
   
   
   if __name__ == '__main__':
       run()
   ```

   在服务端侧，需要实现hello的方法来满足proto文件中GrpcService的接口需求，hello方法的传入参数，是在proto文件中定义的HelloRequest，context是保留字段，不用管，返回参数则是在proto中定义的HelloResponse，服务启动的代码是标准的，可以根据需求修改提供服务的ip地址以及端口号。

5. **编写gRPC客户端的代码**

   ```python
   #! /usr/bin/env python
   # coding=utf8
   import grpc
   
   import hello_pb2_grpc, hello_pb2
   
   
   def run():
       '''
       模拟请求服务方法信息
       :return:
       '''
       conn=grpc.insecure_channel('localhost:50052')
       client = hello_pb2_grpc.GrpcServiceStub(channel=conn)
       skill = hello_pb2.Skill(name="engineer")
       request = hello_pb2.HelloRequest(data="xiao gang", skill=skill)
       respnse = client.hello(request)
       print("received:",respnse.result)
   
   
   if __name__ == '__main__':
       run()
   ```

   客户端侧代码的实现比较简单，首先定义好访问ip和端口号，然后定义好HelloRequest数据结构，远程调用hello即可。需要强调的是，客户端和服务端一定要import相同proto文件编译生成的hello_pb2_grpc, hello_pb2模块，即使服务端和客户端使用的语言不一样，这也是grpc接口规范一致的体现。

6. **调用测试**

   先启动运行服务端的代码，再启动运行客户端的代码即可。

7. **gRPC的使用总结**

   - 定义好接口文档
   - 工具生成服务端/客户端代码
   - 服务端补充业务代码
   - 客户端建立 gRPC 连接后，使用自动生成的代码调用函数
   - 编译、运行