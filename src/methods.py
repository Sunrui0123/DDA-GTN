import torch

def Accuracy_Precision_Sensitivity_Specificity_MCC(args, model, edge_label, edge_label_index, device):
    model.eval()
    with torch.no_grad():
        # z = model.encode()
        # out = model.decode(z, edge_label_index)
        out = model(edge_label_index)
        _, pred = out.max(dim=1)
        model.train()

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index in range(len(pred)):
        if pred[index] == 1 and edge_label[index] == 1:
            TP += 1
        elif pred[index] == 1 and edge_label[index] == 0:
            FP += 1
        elif pred[index] == 0 and edge_label[index] == 1:
            FN += 1
        else:
            TN += 1
    # print('TP: %d, FN: d%, TN: d%, FP: d%' % (TP, FN, TN, FP))
    print('TP=', TP)
    print('FP=', FP)
    print('FN=', FN)
    print('TN=', TN)
    # saving_path = f'result/{args.trainingName}'
    # log_path = saving_path + '/log.txt'
    # result_file = open(file=log_path, mode='w')
    # result_file.write(f'TP = {TP}, FN = {FN},TN = {TN},FP = {FP},\n')
    if (TP + TN + FP + FN) != 0:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        Accuracy = 0
    if (TP + FP) != 0:
        Precision = (TP) / (TP + FP)
    else:
        Precision = 0
    if (TP + FN) != 0:
        Sensitivity = (TP) / (TP + FN)
    else:
        Sensitivity = 0
    if (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) != 0:
        MCC = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    else:
        MCC = 0
    if (FP + TN) != 0:
        Specificity = TN / (FP + TN)
    else:
        Specificity = 0
    return TP, FP, FN, TN






def average_list(list_input):
    average = 0
    for i in range(len(list_input)):
        average = (average * i + list_input[i]) / (i + 1)
    return average



def sum_list(list_input):
    summ = 0
    for i in range(len(list_input)):
        summ = summ + list_input[i]
    return summ