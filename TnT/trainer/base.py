from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy, binary_cross_entropy, softmax
from pathlib import Path
import os, tqdm, torch, numpy as np, shutil, datetime, json
from TnT.utils.transforms import DecodeTarget
import matplotlib.pyplot as plt
from TnT.trainer.metrics import loc_lat_cls_acc

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

class Trainer():
    def __init__(self, lr=1e-4, optim = Adam, sched = CosineAnnealingLR, device='cuda'):
        self.lr = lr
        self.optim = optim
        self.sched = sched
        self.device = device
        
    def _save_train_cfg(self, model, train_dl, val_dl, epochs, early_stop, wdir):
        with open(wdir/'train_cfg.txt', 'w') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Epochs: {epochs}\n")
            if early_stop is not None: f.write(f"Early stopping after {early_stop} epochs of no improvement\n")
            else: f.write(f"Early stopping disabled\n")
            f.write(f"Learningrate: {self.lr}\n")
            f.write(f"Learningrate Scheduler: {self.sched}\n")
            f.write(f"Optimizer: {self.optim}\n")
            f.write(f"Loss: model.loss function\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Train Transforms: {train_dl.dataset.transforms}\n")
            f.write(f"Val Transforms: {val_dl.dataset.transforms}\n")
        
    def train(self, model, train_dl, val_dl, epochs=1, early_stop=5, wdir=Path(datetime.datetime.now().strftime(r'TnTS2_training_from-%H:%M:%S-%d.%m.%y'))):
        os.makedirs(wdir)
        self._save_train_cfg(model, train_dl, val_dl, epochs, early_stop, wdir)
        model.to(self.device)
        self.optim = self.optim(model.parameters(), self.lr)
        self.sched = self.sched(self.optim, epochs)
        loss_history = LossHistory()
        
        fmt = f"0{len(str(epochs))}d"

        for e in range(1, epochs+1):
            print(f'------------------- Epoch {e:{fmt}}/{epochs} -------------------')
            ## Training step
            model.train()
            for batch in tqdm.tqdm(train_dl, desc='Training batches'):
                self.optim.zero_grad()
                l = model.loss(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'], batch['location'].to(self.device))
                l.backward()
                self.optim.step()
                loss_history.add_train(l)
            
            ## Validation step
            model.eval()
            with torch.no_grad():
                for batch in tqdm.tqdm(val_dl, desc='Validating batches'):
                    l = model.loss(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'], batch['location'].to(self.device))
                    loss_history.add_val(l)
            
            ## scheduling
            self.sched.step()
            loss_history.fin_epoch()
            loss_history.plot_progress(wdir)
            
            ## saving
            model.save(wdir/f'epoch_{e}')
            
            ## early stopping
            if loss_history.has_converged(early_stop):
                best_epoch, best_loss = loss_history.min()
                print(f'Convergence achieved after {best_epoch} epochs with validation loss {best_loss}')
                shutil.copytree(wdir/f'epoch_{best_epoch}', wdir/f'best_val_loss')
                with open(wdir/f'best_val_loss'/'note.txt', 'w') as f:
                    f.write(f'Convergence achieved after {best_epoch} epochs with validation loss {best_loss}')
                break
        
        else:
            best_epoch, best_loss = loss_history.min()
            print(f'No convergence achieved after {epochs} epochs. Best loss is {best_loss} at epoch {best_epoch}')
            shutil.copytree(wdir/f'epoch_{best_epoch}', wdir/f'best_val_loss')
            with open(wdir/f'best_val_loss'/'note.txt', 'w') as f:
                f.write(f'No convergence achieved after {epochs} epochs. Best loss is {best_loss} at epoch {best_epoch}')
                
        model.load(wdir/f'best_val_loss')
        return model
                
    def test(self, model, test_dl, best_model_dir=None):
        if best_model_dir is not None:
            model.load(best_model_dir)
        model.to(self.device)
        model.eval()
        decoder = DecodeTarget()
        preds = []
        gts = []
        ids = []
        for batch in tqdm.tqdm(test_dl, desc='Testing batches'):
            ids += batch['id']
            gts.append(batch['location'])
            pred = model.classify(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality']).detach().to('cpu')
            preds.append(pred)
        
        preds = torch.concat(preds, dim=0).to(torch.uint8)  
        preds_dec = decoder(preds)
        gts = torch.concat(gts, dim=0).to(torch.uint8) 
        gts_dec = decoder(gts)
        
        # for id, g, p, g_vec, p_vec in zip(ids, gts_dec, preds_dec, gts, preds):
        #     print(f'Image {id} with GT: loc={g[0]} lat={g[1]} cls={g[2]} got PREDS: loc={p[0]} lat={p[1]} cls={p[2]}')
        #     print(f'    target vector: {g_vec.tolist()}')
        #     print(f'    softmax pred vector: {p_vec.tolist()}')
        
        acc = loc_lat_cls_acc(gts_dec, preds_dec)
        
        print(f"Model achieved an accuracy of {acc}")
        
        return acc