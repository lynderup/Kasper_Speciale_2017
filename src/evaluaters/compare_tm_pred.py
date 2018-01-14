#
# compare_tm_pred.py <true> <pred>
#
# Compares a predicted trans-membrane structure against the true trans-membrane structure
# and computes various statistics summarizing the quality of the prediction. The comparison
# only focuses on the location of the membranes.
#
# The files <true> and <pred> are the true and the predicted structures respectively. The
# files can contain several structures cf. format used in the projects in MLiB Q3/2017.
#
# Christian Storm Pedersen, 09-feb-2017


import sys
#import string
import math

#from fasta import fasta

def fasta(f):
    """
    Reads the fasta file f and returns a dictionary with the sequence names as keys and the
    sequences as the corresponding values. Lines starting with ';' in the fasta file are
    considered comments and ignored.
    """
    d = {}
    curr_key = ""
    lines = [l.strip() for l in open(f).readlines() if (l[0] != ';')]
    for l in lines:
        if l == '': continue
        if l[0] == '>':
            if curr_key != "": d[curr_key] = curr_val
            curr_key = l[1:]
            curr_val = ""
        else:
            curr_val = curr_val + l
    d[curr_key] = curr_val
    
    return d


def count(true, pred):
    tp = fp = tn = fn = 0
    for i in range(len(true)):
        if pred[i] == 'H':
            if true[i] == 'H':
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if true[i] == 'n' or true[i] == 'n':
                tn = tn + 1
            else:
                fn = fn + 1

    return tp, fp, tn, fn

def print_stats(tp, fp, tn, fn, should_print=True):
    sn = sp = cc = acp = float('Inf')
    try:
        sn = float(tp) / (tp + fn)
        sp = float(tp) / (tp + fp)
        cc = float((tp*tn - fp*fn)) / math.sqrt(float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
        acp = 0.25 * (float(tp)/(tp+fn) + float(tp)/(tp+fp) + float(tn)/(tn+fp) + float(tn)/(tn+fn))
    except ZeroDivisionError:
        None
    ac = (acp - 0.5) * 2
    if should_print:
        print("Sn = %.4f, Sp = %.4f, CC = %.4f, AC = %.4f" % (sn, sp, cc, ac))

    return ac, sn, sp


def do_compare(true, pred, should_print):
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

    for key in sorted(true.keys()):
        true_x, true_z = [s.strip() for s in true[key].split('#')]
        pred_x, pred_z = [s.strip() for s in pred[key].split('#')]

        if len(pred_x) != len(pred_z):
            print(len(pred_x))
            print(len(pred_z))
            print("ERROR: prediction on %s has wrong length" % (key))
            sys.exit(1)

        if should_print:
            print(">" + str(key))
        tp, fp, tn, fn = count(true_z, pred_z)
        total_tp, total_fp, total_tn, total_fn = total_tp + tp, total_fp + fp, total_tn + tn, total_fn + fn

        if should_print:
            print_stats(tp, fp, tn, fn)
            print()

    if should_print:
        print("Summary (over all sequences):")
    ac, sn, sp = print_stats(total_tp, total_fp, total_tn, total_fn, should_print)

    return ac, sn, sp


if __name__ == '__main__':
    true = fasta(sys.argv[1])
    pred = fasta(sys.argv[2])
    do_compare(true, pred, True)