import torch
import torch.nn.functional as F  

# def f_score1(inputs, target, beta=0.5, smooth = 1e-5, threhold = 0.5):   #beta=1
#     n, c, h, w = inputs.size()
#     nt, ht, wt, ct = target.size()
#
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
#     temp_inputs_ = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
#     temp_target = target.view(n, -1, ct)
#
#     #--------------------------------------------#
#     #   计算dice系数
#     #--------------------------------------------#
#     temp_inputs = torch.gt(temp_inputs_,threhold).float()
#     _temp_target=torch.less(temp_target,threhold).float()
#     _temp_inputs = torch.less(temp_inputs_, threhold).float()
#     tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
#     fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
#     fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
#     # tn = torch.sum(_temp_inputs, axis=[0, 1]) - fp + tp
#     tn = torch.sum(_temp_target[...,:-1] *_temp_inputs, axis=[0,1])
#
#
#     precision=(tp+smooth)/(tp+fp+smooth)
#     recall=(tp+smooth)/(tp+fn+smooth)
#     score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
#     Accuracy=(tp+tn+smooth)/(tp+tn+fp+fn+smooth)
#
#     # tp=torch.mean(tp)
#     # fp=torch.mean(fp)
#     # fn=torch.mean(fn)
#     # tn=torch.mean(tn)
#
#     score = torch.mean(score)
#     precision=torch.mean(precision)
#     recall=torch.mean(recall)
#     Accuracy=torch.mean(Accuracy)
#
#     return score,precision,recall,Accuracy

def f_score(inputs, target, beta=0.5, smooth=1e-5, threhold=0.5):  # beta=1
    n, c, h, w = inputs.size()
    nt, ht, wt,ct= target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    # temp_inputs_ =torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)


    # --------------------------------------------#
    #   计算dice系数
    # --------------------------------------------#
    # temp_inputs = torch.gt(temp_inputs_, threhold).float()
    # _temp_target = torch.less(temp_target, threhold).float()
    # _temp_inputs = torch.less(temp_inputs_, threhold).float()
    # # a=temp_target[..., :-1]
    # tp = torch.sum(temp_target[..., :-1] * temp_inputs[..., :-1], axis=[0, 1])
    # fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    # fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
    # # tn = torch.sum(_temp_inputs, axis=[0, 1]) - fp + tp
    # tn = torch.sum(_temp_target[..., :-1] * _temp_inputs, axis=[0, 1])

    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
    #
    # # score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    # # score = 1 - torch.mean(score)
    #
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    # Accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    # FPR=(fp+smooth)/(fp+tn+smooth)
    score = torch.mean(score)
    precision = torch.mean(precision)
    recall = torch.mean(recall)
    # Accuracy = torch.mean(Accuracy)
    # fpr=torch.mean(FPR)
    # return score, precision, recall, Accuracy,fpr
    return score, precision, recall
#
