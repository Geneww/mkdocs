

# llama.cpp build



# BLAS Build

Building the program with BLAS support may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). BLAS doesn't affect the normal generation performance. There are currently three different implementations of it:

- #### Accelerate Framework:

  This is only available on Mac PCs and it's enabled by default. You can just build using the normal instructions.

- #### OpenBLAS:

  This provides BLAS acceleration using only the CPU. Make sure to have OpenBLAS installed on your machine.

  - Using `make`:

    - On Linux:

      ```
      make LLAMA_OPENBLAS=1 
      LLAMA_BUILD_SERVER=1
      ```

    - On Windows:

      1. Download the latest fortran version of [w64devkit](https://github.com/skeeto/w64devkit/releases).

      2. Download the latest version of [OpenBLAS for Windows](https://github.com/xianyi/OpenBLAS/releases).

      3. Extract `w64devkit` on your pc.

      4. From the OpenBLAS zip that you just downloaded copy `libopenblas.a`, located inside the `lib` folder, inside `w64devkit\x86_64-w64-mingw32\lib`.

      5. From the same OpenBLAS zip copy the content of the `include` folder inside `w64devkit\x86_64-w64-mingw32\include`.

      6. Run `w64devkit.exe`.

      7. Use the `cd` command to reach the `llama.cpp` folder.

      8. From here you can run:

         ```
         make LLAMA_OPENBLAS=1 LLAMA_BUILD_SERVER=1
         ```

  - Using `CMake` on Linux:

    ```
    mkdir build
    cd build
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS
    cmake --build . --config Release
    ```



server编译

```shell
# windows编译会出现报错 需要添加编译选项
-lwsock32 -lws2_32

make LLAMA_OPENBLAS=1 LLAMA_BUILD_SERVER=1
```

