import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class DynamicKernelGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction_ratio=0.2):
        super(DynamicKernelGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.reduction_ratio = reduction_ratio

        # Channel Kernel Network
        self.channel_kernel_net = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * reduction_ratio)),
            nn.ReLU(),
            nn.Linear(
                int(in_channels * reduction_ratio),
                out_channels * kernel_size * kernel_size,
            ),
        )

        # Spatial Kernel Network
        self.spatial_kernel_net = nn.Conv2d(
            in_channels, kernel_size * kernel_size, kernel_size=1
        )

    def forward(self, x):
        # Channel Kernel
        b, c, h, w = x.size()
        x_pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        channel_kernel = self.channel_kernel_net(x_pooled).view(
            b, self.out_channels, self.kernel_size, self.kernel_size
        )

        # Spatial Kernel
        spatial_kernel = self.spatial_kernel_net(x).view(
            b, h, w, self.kernel_size, self.kernel_size
        )

        # Combine Channel and Spatial Kernels
        dynamic_kernel = channel_kernel.unsqueeze(2).unsqueeze(
            3
        ) * spatial_kernel.unsqueeze(1)
        return dynamic_kernel


class ContextLearningModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextLearningModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class INSTA(nn.Module):
    def __init__(self, backbone, in_channels, out_channels, kernel_size):
        super().__init__()
        self.backbone = backbone
        self.dynamic_kernel_generator = DynamicKernelGenerator(
            in_channels, out_channels, kernel_size
        )
        self.context_learning_module = ContextLearningModule(in_channels, out_channels)

    def forward(self, support_set, query_set):
        # 提取支持集和查询集的特征
        support_features = self.backbone(support_set)
        query_features = self.backbone(query_set)

        # 生成任务特定的 kernel
        task_specific_representation = self.context_learning_module(support_features)
        task_specific_kernel = self.dynamic_kernel_generator(
            task_specific_representation
        )

        # 生成每个支持样本的 instance kernel
        instance_kernels = []
        for i in range(support_features.size(0)):
            instance_kernel = self.dynamic_kernel_generator(
                support_features[i].unsqueeze(0)
            )
            instance_kernels.append(instance_kernel)

        # 融合 instance kernel 和 task-specific kernel
        fused_kernels = [
            instance_kernel * task_specific_kernel
            for instance_kernel in instance_kernels
        ]

        # 对支持集应用动态卷积
        adapted_support_features = []
        for i, kernel in enumerate(fused_kernels):
            adapted_feature = F.conv2d(
                support_features[i].unsqueeze(0), kernel, padding=(kernel.size(-1) // 2)
            )
            adapted_support_features.append(adapted_feature)

        # 对查询集应用 task-specific kernel
        adapted_query_features = F.conv2d(
            query_features,
            task_specific_kernel,
            padding=(task_specific_kernel.size(-1) // 2),
        )

        return adapted_support_features, adapted_query_features


class INSTA_ProtoNet(MetricModel):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.insta = INSTA(self.emb_func, in_channels, out_channels, kernel_size)

    def _forward(self, batch):
        # 解包批量数据
        image, global_target = batch
        image = image.to(self.device)
        # 提取特征
        feat = self.emb_func(image)
        # 将特征划分为支持集和查询集
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        # 计算每个类别的原型
        prototypes = []
        for i in range(self.way_num):
            class_feat = support_feat[:, i * self.shot_num : (i + 1) * self.shot_num, :]
            prototype = class_feat.mean(dim=1)  # 计算类原型
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes, dim=1)  # 形状为 [episode_size, way_num, 特征维度]
        # 计算查询样本与原型之间的距离
        distances = []
        for i in range(self.way_num * self.query_num):
            query = query_feat[:, i, :].unsqueeze(1)  # 形状为 [episode_size, 1, 特征维度]
            distance = torch.norm(
                query - prototypes, dim=2
            )  # 形状为 [episode_size, way_num]
            distances.append(distance)
        distances = torch.stack(
            distances, dim=1
        )  # 形状为 [episode_size, way_num * query_num, way_num]
        # 预测查询样本的类别（距离最小的原型对应的类别）
        output = -distances  # 使用负距离作为 logits
        output = output.view(
            -1, self.way_num
        )  # 形状为 [episode_size * way_num * query_num, way_num]
        # 计算损失和准确率
        acc = accuracy(output, query_target.reshape(-1))
        return output, query_target.reshape(-1), acc

    def set_forward(self, batch):
        o, q, acc = self._forward(batch)
        return o, acc

    def set_forward_loss(self, batch):
        o, q, acc = self._forward(batch)
        return o, acc, F.cross_entropy(o, q)
