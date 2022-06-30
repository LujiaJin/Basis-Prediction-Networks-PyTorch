import torch
import torch.nn as nn
import torch.nn.functional as F


# BPN basic block: SingleConv
class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output


# BPN basic block: DownBlock
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output


# BPN basic block: UpBlock
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output


# BPN basic block: CutEdgeConv
# Used to cut the redundant edge of a tensor after 2*2 Conv2d with valid padding
class CutEdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CutEdgeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2,
                      stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output


# BPN basic block: KernelConv
# Used to predict clean image burst via local convolution
class KernelConv(nn.Module):
    def __init__(self, kernel_size=15):
        super(KernelConv, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, data, kernels):
        """
        compute the pred image according to core and frames
        :param data: [batch_size, burst_length, color_channel, height, width]
        :param kernels: [batch_size, burst_length, kernel_size ** 2, color_channel, height, width]
        :return: pred_burst and pred
        """
        if len(data.size()) == 5:
            batch_size, burst_length, color_channel, height, width = data.size()
        else:
            batch_size, burst_length, height, width = data.size()
            color_channel = 1
            data = data.view(batch_size, burst_length, color_channel, height,
                             width)

        img_stack = []
        kernel_size = self.kernel_size
        data_pad = F.pad(data,
                         [kernel_size // 2, kernel_size // 2, kernel_size // 2,
                          kernel_size // 2])
        for i in range(kernel_size):
            for j in range(kernel_size):
                img_stack.append(data_pad[..., i:i + height, j:j + width])
        img_stack = torch.stack(img_stack, dim=2)
        pred_burst = torch.sum(kernels.mul(img_stack), dim=2, keepdim=False)
        pred = torch.mean(pred_burst, dim=1, keepdim=False)

        return pred_burst, pred


class BPN(nn.Module):
    def __init__(self, color=True, burst_length=8, blind_est=True,
                 kernel_size=15, basis_size=90, upMode='bilinear'):
        super(BPN, self).__init__()
        self.burst_length = burst_length
        self.blind_est = blind_est
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.upMode = upMode
        self.color_channel = 3 if color else 1
        self.in_channel = self.color_channel * (
            self.burst_length if self.blind_est else self.burst_length + 1)
        self.coeff_channel = self.basis_size
        self.basis_channel = self.color_channel * self.burst_length * self.basis_size

        # Layer definition in each block
        # Encoder
        self.initial_conv = SingleConv(self.in_channel, 64)
        self.down_conv1 = DownBlock(64, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        self.down_conv5 = DownBlock(512, 1024)
        self.features_conv1 = SingleConv(1024, 1024)
        self.features_conv2 = SingleConv(1024, 1024)

        # Decoder for coefficients
        self.up_coeff_conv1 = UpBlock(1024 + 1024, 512)
        self.up_coeff_conv2 = UpBlock(512 + 512, 256)
        self.up_coeff_conv3 = UpBlock(256 + 256, 128)
        self.up_coeff_conv4 = UpBlock(128 + 128, 64)
        self.up_coeff_conv5 = UpBlock(64 + 64, 64)
        self.coeff_conv1 = SingleConv(64, 64)
        self.coeff_conv2 = SingleConv(64, 64)
        self.coeff_conv3 = SingleConv(64, self.coeff_channel)
        self.out_coeff = nn.Softmax(dim=1)

        # Decoder for basis
        self.up_basis_conv1 = UpBlock(1024 + 1024, 512)
        self.up_basis_conv2 = UpBlock(512 + 512, 256)
        self.up_basis_conv3 = UpBlock(256 + 256, 256)
        self.up_basis_conv4 = UpBlock(256 + 128, 128)
        self.basis_conv1 = CutEdgeConv(128, 128)
        self.basis_conv2 = SingleConv(128, 128)
        self.basis_conv3 = SingleConv(128, self.basis_channel)
        self.out_basis = nn.Softmax(dim=1)

        # Predict clean images by using local convolutions with kernels
        self.kernel_conv = KernelConv(self.kernel_size)

        # Model weights initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    @staticmethod
    def pad_before_cat(x1, x2):
        """Prevent the image dimensions in the encoder and the decoder from
        being different due to the odd image dimension, which will lead to
        skip concatenation failure."""
        diffY = x1.size()[-2] - x2.size()[-2]
        diffX = x1.size()[-1] - x2.size()[-1]
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x2

    @staticmethod
    def pool_before_cat(data, tosize=2):
        """In the decoder for basis, the features are pooled to 1*1 size and
        then enlarged by replication before skip concatenation."""
        if isinstance(tosize, int):
            height = tosize
            width = tosize
        elif isinstance(tosize, list or tuple) and len(tosize) == 2:
            height, width = tosize
        else:
            raise TypeError(
                "Type error on the parameter 'tosize' that denotes the size "
                "of target tensor. Expect to get an int, or a list/tuple with "
                "the length of 2, but got a {}.".format(
                    type(tosize)))
        pooled_data = F.adaptive_avg_pool2d(data, (1, 1))
        return pooled_data.repeat(1, 1, height, width)

    @staticmethod
    def kernel_predict(coeff, basis, batch_size, burst_length, kernel_size,
                       color_channel):
        """
        return size: (batch_size, burst_length, kernel_size ** 2, color_channel, height, width)
        """
        kernels = torch.einsum('ijklmn,ijop->iklmnop', [basis, coeff]).view(
            batch_size, burst_length, kernel_size ** 2, color_channel,
            coeff.size(-2), coeff.size(-1))
        return kernels

    # forward propagation
    def forward(self, data_with_est, data):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        # input
        initial_conv = self.initial_conv(data_with_est)

        # down sampling
        down_conv1 = self.down_conv1(initial_conv)
        down_conv2 = self.down_conv2(
            F.max_pool2d(down_conv1, kernel_size=2, stride=2))
        down_conv3 = self.down_conv3(
            F.max_pool2d(down_conv2, kernel_size=2, stride=2))
        down_conv4 = self.down_conv4(
            F.max_pool2d(down_conv3, kernel_size=2, stride=2))
        down_conv5 = self.down_conv5(
            F.max_pool2d(down_conv4, kernel_size=2, stride=2))
        features1 = self.features_conv1(
            F.max_pool2d(down_conv5, kernel_size=2, stride=2))
        features = self.features_conv2(features1)

        # up sampling with skip connection, for coefficients
        up_coeff_conv1 = self.up_coeff_conv1(torch.cat([down_conv5,
                                                        self.pad_before_cat(
                                                            down_conv5,
                                                            F.interpolate(
                                                                features,
                                                                scale_factor=2,
                                                                mode=self.upMode))],
                                                       dim=1))
        up_coeff_conv2 = self.up_coeff_conv2(torch.cat([down_conv4,
                                                        self.pad_before_cat(
                                                            down_conv4,
                                                            F.interpolate(
                                                                up_coeff_conv1,
                                                                scale_factor=2,
                                                                mode=self.upMode))],
                                                       dim=1))
        up_coeff_conv3 = self.up_coeff_conv3(torch.cat([down_conv3,
                                                        self.pad_before_cat(
                                                            down_conv3,
                                                            F.interpolate(
                                                                up_coeff_conv2,
                                                                scale_factor=2,
                                                                mode=self.upMode))],
                                                       dim=1))
        up_coeff_conv4 = self.up_coeff_conv4(torch.cat([down_conv2,
                                                        self.pad_before_cat(
                                                            down_conv2,
                                                            F.interpolate(
                                                                up_coeff_conv3,
                                                                scale_factor=2,
                                                                mode=self.upMode))],
                                                       dim=1))
        up_coeff_conv5 = self.up_coeff_conv5(torch.cat([down_conv1,
                                                        self.pad_before_cat(
                                                            down_conv1,
                                                            F.interpolate(
                                                                up_coeff_conv4,
                                                                scale_factor=2,
                                                                mode=self.upMode))],
                                                       dim=1))
        coeff1 = self.coeff_conv1(up_coeff_conv5)
        coeff2 = self.coeff_conv2(coeff1)
        coeff3 = self.coeff_conv3(coeff2)
        coeff = self.out_coeff(coeff3)

        # up sampling with pooled-skip connection, for basis
        up_basis_conv1 = self.up_basis_conv1(torch.cat([self.pool_before_cat(
            down_conv5, tosize=int((self.kernel_size + 1) / 8)), F.interpolate(
            F.adaptive_avg_pool2d(features, (1, 1)), scale_factor=2,
            mode=self.upMode)], dim=1))
        up_basis_conv2 = self.up_basis_conv2(torch.cat([self.pool_before_cat(
            down_conv4, tosize=int((self.kernel_size + 1) / 4)), F.interpolate(
            up_basis_conv1, scale_factor=2, mode=self.upMode)], dim=1))
        up_basis_conv3 = self.up_basis_conv3(torch.cat([self.pool_before_cat(
            down_conv3, tosize=int((self.kernel_size + 1) / 2)), F.interpolate(
            up_basis_conv2, scale_factor=2, mode=self.upMode)], dim=1))
        up_basis_conv4 = self.up_basis_conv4(torch.cat(
            [self.pool_before_cat(down_conv2, tosize=self.kernel_size + 1),
             F.interpolate(up_basis_conv3, scale_factor=2, mode=self.upMode)],
            dim=1))
        basis1 = self.basis_conv1(up_basis_conv4)
        basis2 = self.basis_conv2(basis1)
        basis3 = self.basis_conv3(basis2).view(basis2.size(0),
                                               self.basis_size,
                                               self.burst_length,
                                               self.color_channel,
                                               self.kernel_size,
                                               self.kernel_size)
        basis = self.out_basis(basis3)

        # kernel prediction
        kernels = self.kernel_predict(coeff, basis, coeff.size(0),
                                      self.burst_length, self.kernel_size,
                                      self.color_channel)

        # clean burst prediction
        pred_burst, pred = self.kernel_conv(data, kernels)

        return pred_burst, pred


class LossFunc(nn.Module):
    """
    loss function of BPN
    """

    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True,
                 alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_burst, pred, ground_truth, global_step):
        """
        forward function of loss_func
        :param pred_burst: shape [batch_size, burst_length, color_channel, height, width]
        :param pred: shape [batch_size, color_channel, height, width]
        :param ground_truth: shape [batch_size, color_channel, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred,
                                                  ground_truth), self.coeff_anneal * self.loss_anneal(
            global_step, pred_burst, ground_truth)


class LossBasic(nn.Module):
    """
    Basic loss function.
    """

    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))


class LossAnneal(nn.Module):
    """
    anneal loss function
    """

    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_burst, ground_truth):
        """
        :param global_step: int
        :param pred_burst: [batch_size, burst_length, color_channel, height, width]
        :param ground_truth: [batch_size, color_channel, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_burst.size(1)):
            loss += self.loss_func(pred_burst[:, i, ...], ground_truth)
        loss /= pred_burst.size(1)
        return self.beta * self.alpha ** global_step * loss


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """

    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs(
                (u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow(
                    (u - d)[..., 0:w, 0:h], 2)
            )
