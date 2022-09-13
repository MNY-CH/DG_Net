import torch
import torch.nn as nn
import Network.DGNet as DGNet

class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        num_class = 700
        self.gen = DGNet.DGNet()
        self.teacher = DGNet.ReID(num_class)
        self.id_loss = nn.CrossEntropyLoss()
        self.teacher_loss = nn.KLDivLoss(size_average=False)

    def forward(self, xj, xi, xt):
        Xj_gen, Xi_gen, Xt_gen, Xj_s, Xi_a, Xi_s, Xt_a, Xi_x, Xt_x = self.gen(xj, xi, xt)
        teacher_Xi_a = self.teacher(xi)
        return Xj_gen, Xi_gen, Xt_gen, Xj_s, Xi_a, Xi_s, Xt_a, Xi_x, Xt_x, teacher_Xi_a

    def update(self, xj, xi, xt, xi_gt, xj_gt, Xj_gen, Xi_gen, Xt_gen, Xj_s, Xi_a, Xi_s, Xt_a, Xi_x, Xt_x, teacher_Xi_a):
        Xj_gen_a, Xj_gen_x = self.gen.Ea(Xj_gen)

        loss_recon_1 = self.recon_loss(xi, Xi_gen)
        loss_recon_2 = self.recon_loss(xi, Xt_gen)
        loss_id = self.id_loss(xi_gt, Xi_x)
        loss_cross_recon_1 = self.recon_loss(Xi_a, Xj_gen_a)
        loss_cross_recon_2 = self.recon_loss(Xj_s, self.gen.Es(Xj_gen))
        loss_cross_id = self.id_loss(xj_gt, Xj_gen_x)
        loss_adv = self.gen_loss(self.gen.Discriminator, Xj_gen)
        loss_
        return

    def gen_loss(self, model, generated_img):
        output = model.forward(generated_img)
        loss = 0
        Drift = 0.001

        loss += -torch.mean(output)
        loss += Drift * torch.sum(output ** 2)
        return loss

    def recon_loss(self, input, target):
        return torch.mean(torch.abs(input - target.detach()[:]))