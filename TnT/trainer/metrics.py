from sklearn.metrics import accuracy_score
import numpy as np  
import matplotlib.pyplot as plt 
import json

def loc_lat_cls_acc(gts, preds):
    loc_acc = sum([gt[0]==pred[0] for gt, pred in zip(gts, preds)])/len(gts)
    lat_acc = sum([gt[1]==pred[1] for gt, pred in zip(gts, preds)])/len(gts)
    cls_acc = sum([gt[2]==pred[2] for gt, pred in zip(gts, preds)])/len(gts)
    return {'location acc': loc_acc, 'laterality acc': lat_acc, 'topaneu cls acc': cls_acc}

class LossHistory():
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.cur_val = []
        self.cur_train = []
    
    def add_train(self, l):
        self.cur_train.append(l.item())
        
    def add_val(self, l):
        self.cur_val.append(l.item())
        
    def has_converged(self, patience, tol=1e-4):
        if patience is None: return False
        if len(self.val_loss)<patience: return False
        return not any([f<self.val_loss[-patience] for f in self.val_loss[patience-1:]])
    
    def min(self):
        e, v = min(enumerate(self.val_loss), key=lambda x: x[1])
        return e+1, v
    
    def latest(self):
        e, v = len(self.val_loss), self.val_loss[-1]
        return e, v
    
    def fin_epoch(self):
        self.train_loss.append(sum(self.cur_train)/len(self.cur_train))
        self.cur_train = []
        self.val_loss.append(sum(self.cur_val)/len(self.cur_val))
        self.cur_val = []
        
    def plot_progress(self, wdir):
        plt.plot(self.train_loss, label='training loss')
        plt.plot(self.val_loss, label='validation loss')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.legend()
        plt.title(f'Training/Validation loss per epoch')
        plt.savefig(wdir/'training_losses.png')
        plt.close()
        plt.clf()
        
        with open(wdir/'losses.json', 'w') as f:
            json.dump(
                {
                    'train': self.train_loss,
                    'val': self.val_loss
                },
                f, 
                indent=4
            )
