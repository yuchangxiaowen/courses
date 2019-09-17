# Matlab矩阵计算教程

## 0. 安装

前往[上海财经大学教学实验软件平台](https://jiaoxue.shufe.edu.cn/syzn_420/list.htm)，按照教程进行下载。

## 1. 从Excel读入数据

```matlab
>> A = xlsread('A.xlsx')

A =

     1     2     3
     4     5     6
     7     8     9

>>  b = xlsread('b.xlsx')

b =

     1
     2
     3

```

## 2. 基本运算

Matlab中对应元素运算需要在运算符前加上"."，否则就是进行矩阵运算。
对应元素运算中，若左侧矩阵维度大于右侧，并且左侧矩阵最后几位大小与右侧矩阵本身大小相同，则运算仍然合法，结果为broadcast后的计算，关于broadcast，见"Python矩阵计算教程"中的说明。

```matlab
>> A * b

ans =

    14
    32
    50

>> A .* b

ans =

     1     2     3
     8    10    12
    21    24    27

```

## 3. 其他操作

转置操作

```matlab
>> A.'

ans =

     1     4     7
     2     5     8
     3     6     9

```

矩阵大小
```matlab
>> size(A)

ans =

     3     3
```

创建矩阵
```matlab
>> x = [1, 2; 3, 4]

x =

     1     2
     3     4
```

求和：整体求和、按某个维度求和
```matlab
>> sum(x)

ans =

     4     6

>> sum(x, 1)

ans =

     4     6
```

