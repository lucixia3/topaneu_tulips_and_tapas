from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy, binary_cross_entropy, softmax
from pathlib import Path
import os, tqdm, torch, numpy as np, shutil, datetime, json, copy
import matplotlib.pyplot as plt
from TnT.trainer.base import BasicTrainer
from TnT.trainer.metrics import loc_lat_cls_acc, LossHistory
from TnT.utils.dataloader import TopAneu_TnTs2_DS, DataLoader, TnTs2_collate
from TnT.model.ensemble import Ensemble

class NFoldTrainer():
    def __init__(self, n_folds=2, batch_size=4, lr=1e-4, optim = Adam, sched = CosineAnnealingLR, device='cuda'):
        raise NotImplementedError
        self.lr = lr
        self.optim = optim
        self.sched = sched
        self.device = device
        self.trainer = BasicTrainer
        self.n = n_folds
        self.bs = batch_size
        
    def _save_train_cfg(self, model, epochs, early_stop, wdir):
        with open(wdir/'train_cfg.txt', 'w') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Epochs: {epochs}\n")
            if early_stop is not None: f.write(f"Early stopping after {early_stop} epochs of no improvement\n")
            else: f.write(f"Early stopping disabled\n")
            f.write(f"N folds: {self.n}\n")
            
    def _gen_split_key(self):
        splt = f'{1/self.n}-'*self.n
        return splt.removesuffix('-')
        
    def train(self, model, ds, train_trans, val_trans, epochs=1, early_stop=5, wdir=Path(datetime.datetime.now().strftime(r'TnTS2_Nfold_training_from-%H:%M:%S-%d.%m.%y'))):
        os.makedirs(wdir)
        ds.save(wdir/'full_trainset')
        folds = ds.split(self._gen_split_key())
        for i, f in enumerate(folds):
            f.preprocess()
            f.save(wdir/f'fold_{i}.json')
        
        ensemble = Ensemble(model.n_locs, model.n_lats)
        for hold_out_fold in range(0, self.n):
            fold_wdir = wdir/f"fold_{hold_out_fold}"
            val_ds = folds[hold_out_fold]
            val_ds.transforms = val_trans
            train_ds = TopAneu_TnTs2_DS.join([f for i, f in enumerate(folds) if i!=hold_out_fold])
            train_ds.transforms = train_trans
            cur_trainer = self.trainer(self.lr, self.optim, self.sched, self.device)
            best_model = cur_trainer.train(model=copy.deepcopy(model), train_dl=DataLoader(train_ds, batch_size=self.bs, shuffle=True, collate_fn=TnTs2_collate), val_dl=DataLoader(val_ds, self.bs, shuffle=True, collate_fn=TnTs2_collate), epochs=epochs, early_stop=early_stop, wdir=fold_wdir)
            best_model.save(wdir/f"fold_{hold_out_fold}_best_model")
            ensemble.add(best_model.to('cpu'))
        ensemble.save(wdir/'final_ensemble', overwrite=True)
        return ensemble
    
    def test(self, model, test_dl, best_model_dir=None, decoder=None):
        if best_model_dir is not None:
            model.load(best_model_dir)
        model.to(self.device)
        model.eval()
        preds = []
        gts = []
        ids = []
        for batch in tqdm.tqdm(test_dl, desc='Testing batches'):
            ids += batch['id']
            gts.append(batch['location'])
            pred_lat, pred_loc = model.classify(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'])
            pred_lat=pred_lat.detach().to('cpu')
            pred_loc=pred_loc.detach().to('cpu')
            preds.append(torch.concat([pred_loc, pred_lat], dim=-1))
        
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