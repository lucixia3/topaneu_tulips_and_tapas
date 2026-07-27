from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy, binary_cross_entropy, softmax
from pathlib import Path
import os, tqdm, torch, numpy as np, shutil, datetime, json
from TnT.utils.transforms import DecodeTarget
import matplotlib.pyplot as plt
from TnT.trainer.metrics import loc_lat_cls_acc, LossHistory


class BasicTrainer():
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
                l = model.loss(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'], batch['location'])
                l.backward()
                self.optim.step()
                loss_history.add_train(l)
            
            ## Validation step
            model.eval()
            with torch.no_grad():
                for batch in tqdm.tqdm(val_dl, desc='Validating batches'):
                    l = model.loss(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'], batch['location'])
                    loss_history.add_val(l)
            
            ## scheduling
            self.sched.step()
            loss_history.fin_epoch()
            loss_history.plot_progress(wdir)
            
            ## saving
            model.save(wdir/f'latest_epoch', overwrite=True)
            
            ## saving if best
            if all([l==b for l, b in zip(loss_history.latest(), loss_history.min())]):
                model.save(wdir/'best_val_loss', overwrite=True)
                best_epoch, best_loss = loss_history.min()
                with open(wdir/f'best_val_loss'/'note.txt', 'w') as f:
                    f.write(f'Convergence achieved after {best_epoch} epochs with validation loss {best_loss}')
                
            
            ## early stopping
            if loss_history.has_converged(early_stop):
                break
        
        else:
            best_epoch, best_loss = loss_history.min()
            print(f'No convergence achieved after {epochs} epochs. Best loss is {best_loss} at epoch {best_epoch}')
            shutil.copytree(wdir/f'latest_epoch', wdir/f'best_val_loss', dirs_exist_ok=True)
            with open(wdir/f'best_val_loss'/'note.txt', 'w') as f:
                f.write(f'No convergence achieved after {epochs} epochs. Best loss is {best_loss} at epoch {best_epoch}')
                
        model.load(wdir/f'best_val_loss')
        return model
                
    def test(self, model, test_dl, best_model_dir=None, decoder=DecodeTarget()):
        if best_model_dir is not None:
            model.load(best_model_dir)
        model.to(self.device)
        model.eval()
        preds = []
        gts = []
        ids = []
        for batch in tqdm.tqdm(test_dl, desc='Testing batches'):
            ids += batch['id']
            try: # if its model.stage2_dev
                pred_lat, pred_loc_a, pred_loc_v = model.classify(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'])
                pred_lat=pred_lat.detach().to('cpu')
                pred_loc_v=pred_loc_v.detach().to('cpu')
                preds.append(torch.concat([pred_loc_v, pred_lat], dim=-1))
                gts.append(torch.concat([batch['location']['vessel'], batch['location']['laterality']], dim=-1))
            except: # if its model.stage2
                pred_lat, pred_loc  = model.classify(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'])
                pred_lat=pred_lat.detach().to('cpu')
                pred_loc=pred_loc.detach().to('cpu')
                preds.append(torch.concat([pred_loc, pred_lat], dim=-1))
                gts.append(torch.concat([batch['location']['aneurysm'], batch['location']['laterality']], dim=-1))
        
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