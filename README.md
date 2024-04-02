# Lidar Denoise  轻量级激光雷达点云降噪网络

## 简介
本项目提出了一种新的轻量级网络，用于过滤恶劣天气条件下激光雷达点云中的噪点。该网络采用宽度多级残差模块（WMLR）架构，集成了宽激活、多级残差连接和shuffle attention机制，以实现高效的点云预处理。此外，本项目还介绍了增强型激光雷达数据表示，以提升系统性能。

## 主要内容
车辆的自动驾驶和移动机器人依赖激光雷达传感器进行室外环境感知。但在雾、雨、雪等恶劣天气条件下，空中颗粒物会引入不必要的测量点，导致漏检和误报。传统的雷达点云降噪方法和基于深度学习的方法都存在一定的局限性。为此，本文提出了一种轻量级网络，旨在改善基于激光雷达的感知系统在恶劣天气中的性能。

### 网络架构
本文的网络架构包括三个宽度多级残差模块（WMLR），具有以下特点：
- 宽激活，提供更丰富的特征表示。
- 多级残差连接，增强了网络的学习能力。
- Shuffle attention机制，提高了特征的利用效率。

### 数据表示
提出了增强型激光雷达数据表示，结合了点云空间分布、标准强度和距离输入，提高了性能。

### 模型
提出了两种模型：\aprnet$_2$ 和\aprnet$_3$，它们遵循相同的网络架构但具有不同的输入表示。它们在受控室内环境和自然天气环境下进行训练和测试。

### 性能
通过WADS和Chamber数据集的实验表明，提出的模型性能优于现有的深度学习和传统滤波方法。同时，在计算资源有限的边缘设备上也能够实现最佳的性能和计算成本平衡。

## 表格和图像
以下是实验结果的表格和图像占位符。请根据实际内容添加表格和图像。

### 表格1: Comparison of runtime and model complexity

| Model                          | Params (Mio) | GFLOPs   | Runtime (ms) | Size (M) |
|--------------------------------|--------------|----------|--------------|----------|
| DROR\cite{CRV18DenoisingLidar} | $4e^{-6}$    | -        | 100.00       | $4e^{-6}$|
| LilaNet\cite{LiLaNet18}        | 9.31         | 447.67   | 5.9417       | 37.34    |
| WeatherNet\cite{weatherNet20}  | 1.5313       | 18.2267  | 2.22         | 6.044    |
| PolarNet\cite{PolarNet20}      | 13.6091      | 87.9441  | 22.11        | 54.523   |
| \aprnet$_2$(ours)              | **0.3884**   | **4.9119**| **1.64**     | **1.568**|
| \aprnet$_3$(ours)              | **0.3884**   | \color{blue}{4.9919} | 10.32 | **1.568**|

*Comparison of runtime and model complexity. The best method in **bold** and the second in <span style="color:blue">blue</span>.*

### 表格1: Comparison on the Chamber Dataset

| Model                           | IoU Clear | IoU Fog | IoU Rain | mIoU   | Precision | Recall |
|---------------------------------|-----------|---------|----------|--------|-----------|--------|
| DROR\cite{CRV18DenoisingLidar} | 88.13     | 6.94    | 7.37     | 34.15  | -         | -      |
| LiLanet\cite{LiLaNet18}        | 82.72     | 79.57   | 88.16    | 83.48  | -         | -      |
| WeatherNet\cite{weatherNet20}  | 91.65     | 86.40   | 89.29    | 89.11  | 89.87     | 92.23  |
| PolarNet\cite{PolarNet20}      | **99.08** | **94.75** | 91.35   | 95.06  | **96.33** | 97.00  |
| \aprnet$_2$                    | 97.17     | 91.58   | 94.80    | 94.51  | 95.46     | 97.99  |
| \aprnet$_3$                    | *97.83*   | *93.93* | **95.13**| **95.63**| 93.04    | **98.13**|

*Comparison on the Chamber Dataset. The best method in **bold** and the second best in <span style="color:blue">blue</span>.*


### 表格1: Comparison on the WADS Dataset

| Model                           | IoU Clear | IoU Snow | mIoU   | Precision | Recall |
|---------------------------------|-----------|----------|--------|-----------|--------|
| DROR\cite{CRV18DenoisingLidar} | -         | 67.26    | -      | 71.51     | 91.89  |
| DSOR\cite{DSOR21}              | -         | 63.18    | -      | 65.07     | **95.60** |
| WeatherNet\cite{weatherNet20}  | 92.29     | 81.39    | 86.84  | 94.88     | 85.13  |
| PolarNet\cite{PolarNet20}      | **98.72** | **90.34**| **94.53** | **98.74**| 91.41  |
| \aprnet$_2$                    | 94.73     | 87.31    | 91.02  | 95.42     | 91.13  |
| \aprnet$_3$                    | *95.65*   | *89.53*  | *92.59*| *95.95*   | *93.04*|

*Comparison on the WADS Dataset. The best method in **bold** and the second best in <span style="color:blue">blue</span>.*


### 图像1: 网络架构

![网络架构](path/to/network_architecture.png)

### 图像2: 性能对比

![性能对比](path/to/performance_comparison.png)



