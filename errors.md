### E01
运行 ./configure 时报错 
configure: error: cannot run C compiled programs

**解决方法** ./configure --host=arm

### E02 未解决
安装 PopLDdecay 时运行 Make 报错
ld: symbol(s) not found for architecture x86_64
clang-4.0: error: linker command failed with exit code 1 (use -v to see invocation)
make[1]: *** [Makefile:272: PopLDdecay] Error 1
make[1]: Leaving directory '/Users/sherlock/Downloads/PopLDdecay'
make: *** [Makefile:178: all] Error 2

**解决方法** 未知。或许是在 Makefile 文件修改编译方式，编译方式有 g++, clang, gcc 等。

### E03
Permission denied 
出现原因：没有权限进行读、写、创建文件、删除文件等操作。
**解决方法** 输入命令 sudo chmod -R 777 /工作目录

