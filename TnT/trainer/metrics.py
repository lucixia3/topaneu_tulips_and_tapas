from sklearn.metrics import accuracy_score
import numpy as np  

def loc_lat_cls_acc(gts, preds):
    loc_acc = sum([gt[0]==pred[0] for gt, pred in zip(gts, preds)])/len(gts)
    lat_acc = sum([gt[1]==pred[1] for gt, pred in zip(gts, preds)])/len(gts)
    cls_acc = sum([gt[2]==pred[2] for gt, pred in zip(gts, preds)])/len(gts)
    return {'location acc': loc_acc, 'laterality acc': lat_acc, 'topaneu cls acc': cls_acc}