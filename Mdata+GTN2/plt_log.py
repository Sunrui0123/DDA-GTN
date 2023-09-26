import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


def draw_log(args, epoch, lossTr_list, k, lossVal_list=None):
    # plt loss
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(0, epoch), lossTr_list, label='Train_loss',color='cornflowerblue')
    if lossVal_list != None:
        ax1.plot(range(0, epoch), lossVal_list, label='Val_loss',color='red')
    ax1.set_title("Average training loss vs epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Current loss")
    ax1.legend()
    plt.savefig(args.savedir + "Mdata--"+str(k)+".png")
    plt.close('all')
    plt.clf()

#
# def AUROC(out, data, label):
#     model.eval()
#     p_positive = []
#     output = F.softmax(out)
#     for val in output[:,1]:
#         p_positive.append(val)
#     AUROC_result = roc_auc_score(label, p_positive)
#     fpr, tpr, _ = roc_curve(label, p_positive)
#     precision, recall, _ = precision_recall_curve(label, p_positive)
#     return AUROC_result, fpr, tpr, precision, recall

def return_scores_and_labels(out, label):
    scores = []
    output = F.softmax(out)
    for s in output[:, 1]:
        scores.append(s)
    return scores,label

