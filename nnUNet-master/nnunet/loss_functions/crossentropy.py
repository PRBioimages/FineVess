from torch import nn, Tensor


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print('target.shape',target.shape)  #torch.Size([2, 1, 12, 56, 56])   #从 DC_and_CE_loss入口进去是：[2, 12, 56, 56] 因为经过了target = target[:, 0]
        # print('input.shape',input.shape)   #torch.Size([2, 3, 12, 56, 56])
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1   #assert语句又称作断言，指的是期望用户满足指定的条件。当用户定义的约束条件不满足的时候，它会触发AsserionError异常，所以assert语句可以当作条件式的raise语句。
            target = target[:, 0]
            # print('target',target)  #我加的
            # print('target.shape', target.shape)  #我加的torch.Size([2, 12, 56, 56])
        return super().forward(input, target.long())