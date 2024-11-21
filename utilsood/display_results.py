import numpy as np
import sklearn.metrics as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import os

recall_level_default = 0.95
fpr_level_default = 0.1


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])



def acc_fpr(real_ood_scores, real_in_scores,labels_list, pred_list, fpr_level=recall_level_default):
    # Find ACC@FPRn:
    all_scores = np.concatenate([real_ood_scores, real_in_scores], axis=0)
    all_ood_labels = np.concatenate([np.ones((real_ood_scores.shape[0], 1)), np.zeros((real_in_scores.shape[0], 1))],
                                    axis=0)
    fpr, tpr, thresholds = sk.roc_curve(all_ood_labels.ravel(), all_scores.ravel())

    cutoff = np.argmin(np.abs(fpr - fpr_level))
    score_threshold = thresholds[cutoff]
    print('score_threshold:', score_threshold)
    print('fpr:', fpr[cutoff])
    print('tpr:', tpr[cutoff])
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    in_preds_above_score_threshold = in_preds[in_scores <= score_threshold]  # NOTE: score is MINUS MSP
    in_labels_above_score_threshold = in_labels[in_scores <= score_threshold]
    acc_at_fpr_level = np.mean(in_preds_above_score_threshold == in_labels_above_score_threshold)
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds_above_score_threshold, in_labels_above_score_threshold,
                                                img_num_per_cls, acc_per_cls=True)
    # classwise acc:
    acc_each_class = np.full(num_classes, np.nan)
    for i in range(num_classes):
        _pred = in_preds[in_labels == i]
        _label = in_labels[in_labels == i]
        _N = np.sum(in_labels == i)
        acc_each_class[i] = np.sum(_pred == _label) / _N
    head_acc = np.mean(acc_each_class[0:int(0.5 * num_classes)])
    tail_acc = np.mean(acc_each_class[int(0.5 * num_classes):int(num_classes)])
    acc_str = 'ACC@FPR%s: %.4f (%.4f, %.4f, %.4f | %.4f, %.4f)\n' % (
        fpr_level, acc_at_fpr_level, many_acc, median_acc, low_acc, head_acc, tail_acc)
    # save mask:
    mask = in_scores <= score_threshold
    np.save(os.path.join(save_mask_dir, 'id_sample_coverage_mask_at_fpr%s.npy' % (fpr_level)), mask)





def get_measures(_pos, _neg, in_name, out_name, id_client, use_external, evaluation_score, loss_weight, args, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    ###############################################################
    # 计算ROC曲线
    fpr_plot, tpr_plot, thresholds_plot = roc_curve(labels, examples)
    roc_auc = auc(fpr_plot, tpr_plot)
    plt.figure()
    plt.plot(fpr_plot, tpr_plot, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # file_name_eps = f"{in_name}_{out_name}_{id_client}_{use_external}_{evaluation_score}_{loss_weight}.eps"
    # 保存为EPS文件
    # plt.savefig(file_name_eps, format='eps')
    # plt.show()
    # 生成文件名
    file_name_pkl = f"{in_name}_{out_name}_{id_client}_{use_external}_{evaluation_score}_{loss_weight}_{args.federated}.pkl"
    # 给定的路径
    file_path = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/{file_name_pkl}'

    # 确保路径存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump((fpr_plot, tpr_plot), file)

    ##############################################
    # file_name_eps = f"{in_name}_{out_name}_{id_client}_{use_external}_{evaluation_score}_{loss_weight}.eps"
    # # 指定文件夹路径
    # file_path_eps = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/eps_file/{file_name_eps}'
    #
    # # 确保文件夹存在，如果不存在则创建
    # os.makedirs(os.path.dirname(file_path_eps), exist_ok=True)
    #
    # # 保存图形为.eps文件
    # plt.savefig(file_path_eps, format='eps')

    ###############################################################

    return auroc, aupr, fpr


def show_performance(pos, neg, in_name, out_name, id_client, method_name='ours', recall_level=recall_level_default):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], in_name, out_name, id_client, recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))


def print_measures(auroc, aupr, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t' + method_name)
    print('AUROC{:d} AUPR'.format(int(100*recall_level)))
    print(' & {:.2f} & {:.2f}'.format(100*auroc, 100*aupr))
    #print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    #print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    #print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def print_measures_with_std(aurocs, auprs, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t' + method_name)
    print('AUROC AUPR'.format(int(100*recall_level)))
    print(' & {:.2f} & {:.2f}'.format( 100*np.mean(aurocs), 100*np.mean(auprs)))
    print('& {:.2f} & {:.2f}'.format(100*np.std(aurocs), 100*np.std(auprs)))
    #print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    #print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    #print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))


def show_performance_comparison(pos_base, neg_base, pos_ours, neg_ours, in_name, out_name, id_client, baseline_name='Baseline',
                                method_name='ours', recall_level=recall_level_default):
    '''
    :param pos_base: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg_base: 0's class scores generated by the baseline
    '''
    auroc_base, aupr_base, fpr_base = get_measures(pos_base[:], neg_base[:], in_name, out_name, id_client, recall_level)
    auroc_ours, aupr_ours, fpr_ours = get_measures(pos_ours[:], neg_ours[:], in_name, out_name, id_client, recall_level)

    print('\t\t\t' + baseline_name + '\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
        int(100 * recall_level), 100 * fpr_base, 100 * fpr_ours))
    print('AUROC:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * auroc_base, 100 * auroc_ours))
    print('AUPR:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * aupr_base, 100 * aupr_ours))
    # print('FDR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
    #     int(100 * recall_level), 100 * fdr_base, 100 * fdr_ours))
