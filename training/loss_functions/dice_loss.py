#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
from batchgenerators.augmentations.utils import resize_segmentation

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., apply_nonlin=None, batch_dice=False, do_bg=True, smooth_in_nom=True,
                 background_weight=1, rebalance_weights=None, square_nominator=False, square_denom=False):
        """
        hahaa no documentation for you today
        :param smooth:
        :param apply_nonlin:
        :param batch_dice:
        :param do_bg:
        :param smooth_in_nom:
        :param background_weight:
        :param rebalance_weights:
        """
        super(SoftDiceLoss, self).__init__()
        self.square_denom = square_denom
        self.square_nominator = square_nominator
        if not do_bg:
            assert background_weight == 1, "if there is no bg, then set background weight to 1 you dummy"
        self.rebalance_weights = rebalance_weights
        self.background_weight = background_weight
        if smooth_in_nom:
            self.smooth_in_nom = smooth
        else:
            self.smooth_in_nom = 0
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.y_onehot = None

    def forward(self, x, y):
        with torch.no_grad():
            y = y.long()
        shp_x = x.shape
        # print('x shape is:',shp_x)
        shp_y = y.shape
        # print('y shape is:',shp_y)
        #y shape is: torch.Size([8, 1, 192, 192, 48])
        # nonlin maybe mean NONLINEARITY!
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        # print('After apply nonlin, x shape is:',x.shape)
        #After apply nonlin, x shape is: torch.Size([8, 3, 192, 192, 48])
        # output shape is: [8,3,192,192,48] when batch size is 8 and labels are [0,1,2]
        # now x and y should have shape (B, C, X, Y(, Z))) and (B, 1, X, Y(, Z))), respectively
        y_onehot = torch.zeros(shp_x)
        if x.device.type == "cuda":
            y_onehot = y_onehot.cuda(x.device.index)
        y_onehot.scatter_(1, y, 1)
        if not self.do_bg:
            x = x[:, 1:]# This means to reduce the first 0 dimension of the shape of output x, to remove background prediction
            # x is the probability output, so its range is between [0,1]
            y_onehot = y_onehot[:, 1:]
        # print('y_onehot shape is:',y_onehot.shape)
        # y_onehot shape is: torch.Size([8, 2, 192, 192, 48])
        # print('The last version of x shape is:',x.shape)
        #The last version of x shape is: torch.Size([8, 2, 192, 192, 48])
        # print('x max is:', torch.max(x))
        # x max is: tensor(1.0000, device='cuda:4', grad_fn=<MaxBackward1>)
        # print('x min is:', torch.min(x))
        # x min is: tensor(3.1973e-07, device='cuda:4', grad_fn=<MinBackward1>)
        # print('y_onehot max is:', torch.max(y_onehot))
        #y_onehot max is: tensor(1., device='cuda:4')
        #x max is: tensor(1.0000, device='cuda:4', grad_fn=<MaxBackward1>)
        # print('y_onehot min is:', torch.min(y_onehot))
        #y_onehot min is: tensor(0., device='cuda:4')


        if not self.batch_dice:
            if self.background_weight != 1 or (self.rebalance_weights is not None):
                raise NotImplementedError("nah son")

            l = soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom, self.square_nominator, self.square_denom)
            # print('Using soft_dice!')
        else:

            l = soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom,
                                      background_weight=self.background_weight,
                                      rebalance_weights=self.rebalance_weights)
            # print('Using soft_dice_per_batch_2!')
            # Here we use the soft_dice_per_batch_2
        # print('dc shape is:',l.size())
        # dc_loss is: tensor(-0.0282, device='cuda:4', grad_fn=<MeanBackward0>)
        return l


def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1, rebalance_weights=None,
                        square_nominator=False, square_denom=False):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:] # this is the case when use_bg=False
    # print('\nrebalance_weights is:',rebalance_weights)
    #rebalance_weights is: None
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    # print('\naxes is:',axes)
    #axes is: (0, 2, 3, 4)
    # print('\nnet_output shape is:',net_output.shape)
    # print('\ngt shape is:',gt.shape)
    #net_output shape is: torch.Size([8, 2, 192, 192, 48])
    #gt shape is: torch.Size([8, 2, 192, 192, 48])
    tp = sum_tensor(net_output * gt, axes, keepdim=False)
    # print('\ntp is:',tp)
    #tp shape is: torch.Size([2])
    #tp is: tensor([62684.4570, 82510.1562], device='cuda:4', grad_fn=<SumBackward2>)
    fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    # print('\nfn is:',fn)
    #fn shape is: torch.Size([2])
    #fn is: tensor([195664.5312, 103144.8438], device='cuda:4', grad_fn=<SumBackward2>)
    fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    # print('\nfp is:',fp)
    #fp shape is: torch.Size([2])
    #fp is: tensor([3610596., 6475380.], device='cuda:4', grad_fn=<SumBackward2>)
    weights = torch.ones(tp.shape)
    # print('\nweights shape is:',weights.shape)
    #weights shape is: torch.Size([2])
    weights[0] = background_weight
    # print('\nbackground_weight is:',background_weight)
    #background_weight is: 1
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float()
        if net_output.device.type == "cuda":
            rebalance_weights = rebalance_weights.cuda(net_output.device.index)
        tp = tp * rebalance_weights
        fn = fn * rebalance_weights

    nominator = tp

    if square_nominator:
        nominator = nominator ** 2

    if square_denom:
        denom = 2 * tp ** 2 + fp ** 2 + fn ** 2
    else:
        denom = 2 * tp + fp + fn

    # result_1=(- ((2 * nominator + smooth_in_nom) / (denom + smooth)) * weights)
    # print('\nresult_1 is:',result_1)
    #result_1 is: tensor([-0.0616, -0.0038], device='cuda:4', grad_fn=<MulBackward0>)
    dice_1 = ( ((2 * nominator + smooth_in_nom) / (denom + smooth)) * weights)
    # print('\ndice_1 is:',dice_1)
    result_1=torch.pow((-torch.log(dice_1[0])),0.3)*0.4+torch.pow((-torch.log(dice_1[1])),0.3)*0.6
    # print('\nresult_1 is:',result_1)


    # result = (- ((2 * nominator + smooth_in_nom) / (denom + smooth)) * weights).mean()
    # print('\nresult is:',result)
    # result is: tensor(-0.0327, device='cuda:4', grad_fn= < MeanBackward0 >)
    # Here we should notice that the soft dice is set as negative.
    return result_1


def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1., square_nominator=False, square_denom=False):
    axes = tuple(range(2, len(net_output.size())))
    if square_nominator:
        intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    else:
        intersect = sum_tensor((net_output * gt) ** 2, axes, keepdim=False)
    if square_denom:
        denom = sum_tensor(net_output ** 2 + gt ** 2, axes, keepdim=False)
    else:
        denom = sum_tensor(net_output + gt, axes, keepdim=False)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()
    return result




class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        # print('target shape is:',target.shape)
        #target shape is: torch.Size([8, 1, 192, 192, 48])
        ce_weights = torch.tensor([0.28, 0.28, 0.44]).to(torch.cuda.current_device())
        ce_1 = CrossentropyND(weight=ce_weights)
        # dc_loss = self.dc(net_output, target)
        # # ce_loss = self.ce(net_output, target)
        # ce1_loss = ce_1(net_output, target)
        # target_layers=list()
        dc_loss_layers=list()
        ce_loss_layers=list()

        if isinstance(target, list):
            # print('The target is list!')

            for i in range(len(target)):
                # print('net_output[%d] is cuda?'%(2*i),net_output[2*i].is_cuda)
                # print('target[%d] is cuda?' % (i), target[i].is_cuda)
                # print('target %d shape is:'%i,target[i].shape)
                # print('net_output %d shape is:'%(2*i),net_output[2*i].shape)
                # print('net_output[%d] shape is:'%i,net_output[i].shape)
                # print('target[%d] shape is:' % i, target[i].shape)
                dc_loss_layers.append(self.dc(net_output[i], target[i]))
                ce_loss_layers.append(ce_1(net_output[i], target[i]))

            dc_loss=dc_loss_layers[0]*0.6+dc_loss_layers[1]*0.1+dc_loss_layers[2]*0.1+dc_loss_layers[3]*0.1+dc_loss_layers[4]*0.1
            ce_loss=ce_loss_layers[0]*0.6+ce_loss_layers[1]*0.1+ce_loss_layers[2]*0.1+ce_loss_layers[3]*0.1+ce_loss_layers[4]*0.1
            # print('Final dc_loss is:',dc_loss)
            # print('Final ce_loss is:',ce_loss)
            if self.aggregate == "sum":
                result = ce_loss + dc_loss

            else:
                raise NotImplementedError("nah son") # reserved for other stuff (later)
            return result
        else:
            # print('Target is not list!')
            dc_loss = self.dc(net_output, target)
            ce_loss = ce_1(net_output, target)
            if self.aggregate == "sum":
                result = ce_loss + dc_loss

            else:
                raise NotImplementedError("nah son")  # reserved for other stuff (later)
            return result

