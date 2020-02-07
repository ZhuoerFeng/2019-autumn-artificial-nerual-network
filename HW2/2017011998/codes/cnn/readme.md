real_data,data文件夹下为实验数据，6个list分别为training acc/loss, validation acc/loss, testing acc/loss，他们的区别是，data目录下的数据未优化CNN，而real_data的数据CNN达到最优情况。
为了输出数据，我在main函数中增加了数个record list容器，具体在75行前后
