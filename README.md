# VGG16-CNN-image-classification
机器视觉课程作业

VGG论文出处：Very Deep Convolutional Networks for Large-Scale Image Recognition

site：https://paperswithcode.com/paper/very-deep-convolutional-networks-for-large

数据集地址：https://www.cs.toronto.edu/~kriz/cifar.html

预训练权值地址：https://download.pytorch.org/models/vgg11_bn-6002323d.pth

运行：1.在Anaconda下创建虚拟环境 -- conda create VGG

​			2.激活环境 -- conda activate VGG

​			3.安装依赖：pip install requirements.txt

​			4.终端执行:python vgg16_cifar-10_base.py

​							  python vgg16_cifar-10_transfer.py

需要cuda version :12.1  , torch version >2.0.0

说明:1.vgg16_cifar-10_base.py实现VGG16-net，若使用CPU训练可能耗时过长

​		2.vgg16_cifar-10_transfer.py实现带有预训练权重的VGG16-net，运行效率以及准确率都有明显提升，但需要注意内存以及显存的占用量，有可能出现爆内存的情况
