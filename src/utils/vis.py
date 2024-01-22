import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# visulization class

class OSMVisulizer:
    def __init__(self, mapping, threshold=0.7, path="./"):
        self.name = "OSMVisulizer"
        self.channel_to_rgb = mapping
        self.threshold = threshold
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)

    def minmax(self, data):
        data = data - data.min()
        data = data / data.max()
        return data

    def visulize_onehot_layout(self, data, path=None) -> None:
        b, c, h, w = data.shape
        # print(data.shape)
        # exit(0)
        # print(path)
        assert (
            b > 1
        ), f"batch size {b} is too small, please set batch size larger than 1"

        fig, axes = plt.subplots(b, c, figsize=(20, 20))

        for i in range(b):
            for j in range(c):
                if c == 1:
                    axes[i].imshow(data[i, j, :, :].cpu().numpy(), cmap="gray")
                else:
                    axes[i, j].imshow(data[i, j, :, :].cpu().numpy(), cmap="gray")
                    axes[i, j].axis("off")
        if path is None:
            path = os.path.join(self.path, "onehot.png")
        plt.savefig(path)
        plt.close()

    def hex_or_name_to_rgb(self, color):
        # 使用matplotlib的颜色转换功能
        return mcolors.to_rgb(color)

    # need test
    # input must be a one hot layout
    def visualize_rgb_layout(self, data, path=None) -> np.ndarray:
        B, C, H, W = data.shape
        # assert (
        #     self.channel_to_rgb.__len__() >= C
        # ), f"channel to rgb mapping length {self.channel_to_rgb.__len__()} does not match channel number {C}"

        assert (
            B > 1
        ), f"batch size {B} is too small, please set batch size larger than 1"
        data = data.cpu().numpy()

        fig, axs = plt.subplots(B, 1, figsize=(20, 20 * B))  # 根据批次数量创建子图

        for b in range(B):
            combined_image = np.zeros((H, W, 3), dtype=np.float32)
            for c in range(C):
                if c >= self.channel_to_rgb.__len__():
                    break # 超出颜色映射表的通道不再绘制
                color = np.array(self.hex_or_name_to_rgb(self.channel_to_rgb[c])) # haddle condition result
                mask = data[b, c] > self.threshold
                combined_image[mask, :] += color

            combined_image = np.clip(combined_image, 0, 1)  # 确保颜色值在0-1范围内
            axs[b].imshow(combined_image)
            axs[b].axis("off")  # 关闭坐标轴

        # 绘制图像
        plt.axis("off")
        plt.imshow(combined_image)
        plt.show()
        if path is None:
            path = os.path.join(self.path, "rgb.png")
        plt.savefig(path)
        plt.close()

        # return shape : (b, h, w, c)
        return None

    def onehot_to_rgb(self, data):
        B, C, H, W = data.shape
        assert (
            self.channel_to_rgb.__len__() >= C
        ), f"channel to rgb mapping length {self.channel_to_rgb.__len__()} does not match channel number {C}"

        combined_image = torch.zeros(
            (B, H, W, 3), dtype=torch.float32, device=data.device
        )

        for b in range(B):
            for c in range(C):
                color = torch.tensor(
                    self.hex_or_name_to_rgb(self.channel_to_rgb[c]), device=data.device
                )
                mask = data[b, c] > self.threshold
                combined_image[b, mask, :] += color

        combined_image = self.minmax(
            combined_image.permute(0, 3, 1, 2)
        )  # normalize to 0-1

        if torch.isnan(combined_image).any():
            combined_image[torch.isnan(combined_image)] = 0

        return combined_image