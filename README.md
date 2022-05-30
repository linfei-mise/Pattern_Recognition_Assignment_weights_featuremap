## 该文件夹是用来存放模式识别作业中卷积神经网络生成特征图和权重直方图的脚本
### 介绍
* （1）针对应用的alexnet、resnet、convnext三种卷积神经网络进行feature_map和weights_map的提取。
* （2）执行时需要导入对应网络的模型文件。
* （3）结果示例请见对应的文件夹。
### 文件类型介绍：
```
├── analyze_feature_map.py（生成特征图所应用的脚本）  
└── analyze_kernel_weight.py（生成直方图所应用的脚本） 
```
### 信息及维护时间：
```
作者：林飞
创建：2022-05-17
更新：2022-05-22
用途：获取卷积神经网络的feature_map和weights_map。
```
