import torch
import torch.nn as nn
from functools import partial
from torch.autograd import Variable
from distance import compute_distance_matrix


class MaximumMeanDiscrepancy(nn.Module):

    """
    Implementation of MMD :
    https://github.com/shafiqulislamsumon/HARTransferLearning/blob/master/maximum_mean_discrepancy.py
    """

    def __init__(self, batch_size=32, instances=4, mode="UDA"):
        super(MaximumMeanDiscrepancy, self).__init__()
        self.batch_size = batch_size
        self.instances = instances
        self.mode = mode

    # Consider linear time MMD with a linear kernel:
    # K(f(x), f(y)) = f(x)^Tf(y)
    # h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
    #             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
    #
    # f_of_X: batch_size * k
    # f_of_Y: batch_size * k
    def mmd_linear(self, f_of_X, f_of_Y):
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params: 
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            sum(kernel_val): 多个核矩阵之和
        """
        # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        n_samples = int(source.size()[0]) + int(target.size()[0])
        # 将source,target按列方向合并
        # [256, 8]
        total = torch.cat([source, target], dim=0)
        # 将total复制（n+m）份
        # [256, 256, 8]
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        # [256, 256, 8]
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        # [256, 256]
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # 调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        # 高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # /len(kernel_val)

    def mmd_rbf_accelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma
        )
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        return loss / float(batch_size)

    def mmd_rbf_noaccelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def pairwise_distance(self, x, y):
        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError("Both inputs should be matrices.")

        if x.shape[1] != y.shape[1]:
            raise ValueError("The number of features should be the same.")

        # [128, 8, 1]
        # broadcast:
        x = x.view(x.shape[0], x.shape[1], 1)
        # [8, 128]
        y = torch.transpose(y, 0, 1)
        # [128 ,128]
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)
        return output

    def gaussian_kernel_matrix(self, x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1.0 / (2.0 * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_)
        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(self, x, y, kernel=gaussian_kernel_matrix):
        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))
        return cost

    def mmd_loss(self, source, target):
        sigmas = [
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            100,
            1e3,
            1e4,
            1e5,
            1e6,
        ]
        gaussian_kernel = partial(
            self.gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
        )
        loss_value = self.maximum_mean_discrepancy(source, target, kernel=gaussian_kernel)
        return loss_value

    def forward(self, source_features, target_features):
        # group each images of the same identity together
        instances = self.instances
        batch_size = self.batch_size
        num_spk = int(batch_size / instances)
        feature_size = target_features.shape[1]  # 2048

        # [128, 128]
        s_dist = compute_distance_matrix(source_features.squeeze(2), source_features.squeeze(2))

        # wcs = torch.triu(s_dist[:instances, :instances], diagonal=1)
        # bcs = s_dist[:instances, instances:2*instances]
        # for j in range(2, num_spk):
        #     bcs = torch.cat((bcs, s_dist[:instances, j*instances: (j+1)*instances]))
        # for i in range(1, num_spk):
        #     wcs_tmp = torch.triu(s_dist[i*instances: (i+1)*instances, i*instances: (i+1)*instances], diagonal=1)
        #     wcs = torch.cat((wcs, wcs_tmp))
        #     # for j in range(num_spk):
        #     #     if i != j:
        #     for j in range(i+1, num_spk):
        #         bcs = torch.cat((bcs, s_dist[i*instances: (i+1)*instances, j*instances: (j+1)*instances]))
        # bcs = bcs.flatten().unsqueeze(1)
        # wcs = wcs.flatten().unsqueeze(1)
        # wcs = torch.stack([x for x in wcs if x[0] > 0.0])
        wcs = s_dist[:instances, :instances]
        bcs = s_dist[:instances, instances : 2 * instances]
        for j in range(2, num_spk):
            bcs = torch.cat((bcs, s_dist[:instances, j * instances : (j + 1) * instances]))
        for i in range(1, num_spk):
            wcs = torch.cat(
                (
                    wcs,
                    s_dist[
                        i * instances : (i + 1) * instances, i * instances : (i + 1) * instances
                    ],
                )
            )
            for j in range(num_spk):
                if i < j:
                    bcs = torch.cat(
                        (
                            bcs,
                            s_dist[
                                i * instances : (i + 1) * instances,
                                j * instances : (j + 1) * instances,
                            ],
                        )
                    )

        t_dist = compute_distance_matrix(target_features.squeeze(2), target_features.squeeze(2))
        # wct = t_dist[:instances, :instances]
        # wct = torch.triu(t_dist[:instances, :instances], diagonal=1)
        # bct = t_dist[:instances, instances:2*instances]
        # for j in range(2, num_spk):
        #     bct = torch.cat((bct, t_dist[:instances, j*instances: (j+1)*instances]))
        # for i in range(1, num_spk):
        #     wct_tmp = torch.triu(t_dist[i*instances: (i+1)*instances, i*instances: (i+1)*instances], diagonal=1)
        #     wct = torch.cat((wct, wct_tmp))
        #     # for j in range(num_spk):
        #     #     if i != j:
        #     for j in range(i+1, num_spk):
        #         bct = torch.cat((bct, t_dist[i*instances: (i+1)*instances, j*instances: (j+1)*instances]))
        # bct = bct.flatten().unsqueeze(1)
        # wct = wct.flatten().unsqueeze(1)
        # wct = torch.stack([x for x in wct if x[0] > 0.0])
        wct = t_dist[:instances, :instances]
        bct = t_dist[:instances, instances : 2 * instances]
        for j in range(2, num_spk):
            bct = torch.cat((bct, t_dist[:instances, j * instances : (j + 1) * instances]))
        for i in range(1, num_spk):
            wct = torch.cat(
                (
                    wct,
                    t_dist[
                        i * instances : (i + 1) * instances, i * instances : (i + 1) * instances
                    ],
                )
            )
            for j in range(num_spk):
                if i < j:
                    bct = torch.cat(
                        (
                            bct,
                            t_dist[
                                i * instances : (i + 1) * instances,
                                j * instances : (j + 1) * instances,
                            ],
                        )
                    )

        if self.mode in ["UDA", "SDA"]:
            return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct)
            # return self.mmd_rbf_noaccelerate(wcs, wct), self.mmd_rbf_noaccelerate(bcs, bct)
        elif self.mode in ["CUDA", "CSDA"]:
            return (
                self.mmd_loss(wcs, wct),
                self.mmd_loss(bcs, bct),
                self.mmd_loss(wcs, bct),
                self.mmd_loss(bcs, wct),
            )
            # return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct), \
            #         self.mmd_loss(wcs, bcs), self.mmd_loss(wct, bct)
        elif self.mode in ["hard_sample"]:
            wcs_hard = wcs[torch.topk(torch.sum(wcs, dim=1), k=int(wcs.size(0) * 0.20))[1]]
            wct_hard = wct[torch.topk(torch.sum(wct, dim=1), k=int(wct.size(0) * 0.20))[1]]
            bcs_hard = bcs[torch.topk(torch.sum(bcs, dim=1), k=int(bcs.size(0) * 0.20), largest=False)[1]]
            bct_hard = bct[torch.topk(torch.sum(bct, dim=1), k=int(bct.size(0) * 0.20), largest=False)[1]]
            return (
                self.mmd_loss(wcs, wct),
                self.mmd_loss(bcs, bct),
                self.mmd_loss(wcs_hard, wct_hard),
                self.mmd_loss(bcs_hard, bct_hard),
                self.mmd_loss(wcs, bct),
                self.mmd_loss(bcs, wct),
            )
        elif self.mode in ["emb_mmd"]:
            return self.mmd_loss(source_features.squeeze(2), target_features.squeeze(2))


if __name__ == "__main__":
    mmd = MaximumMeanDiscrepancy(mode="emb_mmd")
    torch.manual_seed(1024)
    device = torch.device("cuda:0")
    source = torch.randn(4096, 1, 1).to(device)
    source = source * 2
    source = source + 3.1
    target = torch.randn(256, 1, 1).to(device)
    result = mmd(source, target)
    print(result)
