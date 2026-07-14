from pprint import pprint
from pathlib import Path
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader
from TnT.utils.transforms import AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel
from TnT.model.stage2 import TnTS2
from TnT.trainer.base import Trainer

if __name__ == '__main__':
    ## Do splits
    # ds = TopAneu_TnTs2_DS("/home/tue20260926/Data/topaneu_deployment")
    # folds = ds.split('0.8-0.1-0.1', 42)
    # for id, fold in zip(['train', 'test', 'val'], folds):
    #     fold.preprocess()
    #     fold.save(id)
    
    ## Load splits
    transforms = Compose([
        MaybeToTensor(),
        AdaNorm.make(),
        MaybeResize(size=64),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
    ])
    train = TopAneu_TnTs2_DS.load('train.json', transforms)
    val = TopAneu_TnTs2_DS.load('val.json', transforms)
    test = TopAneu_TnTs2_DS.load('test.json', transforms)
    
    train_dl = DataLoader(train, batch_size=4, shuffle=True)
    val_dl = DataLoader(val, batch_size=4, shuffle=True)
    test_dl = DataLoader(test, batch_size=4, shuffle=False)
    
    trainer = Trainer()
    model = TnTS2()
    model.load('TnTS2_training_from-14:51:13-14.07.26/best_val_loss')
    
    model = trainer.train(model, train_dl, val_dl, 2, None)
    acc = trainer.test(model, test_dl)