## 作业提交

计算任务是通过脚本文件提交到作业管理系统的，脚本文件是一个常规文本文件，具有执行权限，可以直接在登入节点使用vi 编辑器编写，也可异地编写上传至用户作业工作目录，但要注意dos2unix 转换一下。

脚本文件名无特殊规定，起一个可识别的名字即可。编辑完成脚本文件后，将脚本赋予可执行权限，然后提交。例如对一个名称为vasp.lsf 的作业脚本文件，编辑完成后，需要执行命令`chmod +x vasp.lsf`赋予执行权限，然后使用命令`bsub <./vasp.lsf`来提交。注意，提交命令中，在`vasp.lsf`前面有`./`指定`vasp.lsf`脚本的位置。
作业脚本范例：以vasp.sh 举例

```
#!/bin/bash
#BSUB -cwd .（作业的当前目录）
#BSUB -J test   (任务名称） 
#BSUB -e %J.err  (错误标准输出 )
#BSUB -o %J.out  (标准输出文件 )
#BSUB -q gauss01(提交的队列 )
#BSUB -R span[ptile=12] (每台机器使用的CPU core)
#BSUB -n 24 (提交任务所需要的core数 )

./vasp.lsf
```

[^1]: [LSF官方文档](https://www.ibm.com/support/knowledgecenter/SSWRJV/product_welcome_spectrum_lsf.html)

[^2]: [GPFS官方文档](https://www.ibm.com/support/knowledgecenter/STXKQY/ibmspectrumscale_welcome.html)
