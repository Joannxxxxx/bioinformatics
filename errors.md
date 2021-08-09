### E01
运行 ./configure 时报错 
configure: error: cannot run C compiled programs

**解决方法** ./configure --host=arm

### E02 
安装 PopLDdecay 时运行 Make 报错<br>
ld: symbol(s) not found for architecture x86_64<br>
clang-4.0: error: linker command failed with exit code 1 (use -v to see invocation)<br>
make[1]: *** [Makefile:272: PopLDdecay] Error 1<br>
make[1]: Leaving directory '/Users/sherlock/Downloads/PopLDdecay'<br>
make: *** [Makefile:178: all] Error 2<br>

**解决方法** 
未知。或许是在 Makefile 文件修改编译方式，编译方式有 g++, clang, gcc 等。<br>

20210809 已解决。安装 LDBlockShow 出现同样问题，在开发者指导下，<br>
首先在 src 文件夹 make.sh 第 30 行将编译方式 g++ 改为 g++ -std=c++11 或者 clang++，<br>
然后将 14、15 行的路径 src/ 改为 ./ （即当前路径），<br>
最后运行 sh make.sh 即可<br>

测试是否安装成功：<br>
切换文件路径至 Example2，运行 sh run.sh 看结果<br>



### E03
Permission denied<br>
出现原因：没有权限进行读、写、创建文件、删除文件等操作。<br>

**解决方法** 输入命令 sudo chmod -R 777 /工作目录

