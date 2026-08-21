import json
from copy import deepcopy
from collections import Counter
from pprint import pprint
from TnT.utils.dataloader import TopAneu_TnTs2_DS
with open('/home/tue20260926/Repos/TopAneu-26/topaneu_release/location_mapping.json', 'r') as f:
            mapping = {v:k for k, v in json.load(f)['labels'].items()}
file  = '/home/tue20260926/Repos/topaneu_tulips_and_tapas/results_perfseg_tta/classification_failure.json'
print('#################### Expoloring Results ####################')
print(' File:', file)
te = TopAneu_TnTs2_DS.load('tuning-test.json', None)
labels_te = []
for i in range(len(te)):
    labels_te.append(te[i]['location_a'])
n_aneu_gt = (len(te))
labels_te = {mapping[k]:v for k, v in Counter(labels_te).items()}

with open(file, 'r') as f:
    cls_fail = json.load(f)
n_failed = len(cls_fail)
print(' N-Failed-Aneurysms:', n_failed, f'({n_failed/n_aneu_gt:.4f}%)')

tr = TopAneu_TnTs2_DS.load('train.json', None)
tr.preprocess(include_bg=False, include_pred=False, include_syn=False, max_items=-1, filter=None)
        
labels = []
for i in range(len(tr)):
    labels.append(tr[i]['location_a'])
n_in_train = (len(tr))
labels = {mapping[k]:v for k, v in Counter(labels).items()}

failed_gts = []
failure_modes = {'other':[]}
for name, fail in cls_fail.items():
    gt = fail['GT']
    pred = fail['Predicted']
    failed_gts.append(gt)
    grp_gt = deepcopy(gt).split('.')[0][-1]
    grp_pred = deepcopy(pred).split('.')[0][-1]
    
    ## L-R failure
    if gt[1]=='-' and pred[1]=='-' and gt[2:]==pred[2:]:
        if 'L-R missmatch' not in failure_modes.keys(): failure_modes['L-R missmatch']=1
        else: failure_modes['L-R missmatch']+=1
    
    ## Correct group, incorrect subgroup
    elif grp_gt == grp_pred:
        if 'Within-group subcategory missmatch' not in failure_modes.keys(): failure_modes['Within-group subcategory missmatch']=1
        else: failure_modes['Within-group subcategory missmatch']+=1
        
    else:
        failure_modes['other'].append(name)
        
cnt_fgt = Counter(failed_gts)
print('Failed cases and their occurance in the training data:')
for k,v in cnt_fgt.items():
    print(f'    {k}:\n        > In Test: {v} ({labels_te[k]/n_aneu_gt if k in labels_te.keys() else 0}%)\n        > In Train: {labels[k] if k in labels.keys() else 0} ({labels[k]/n_in_train if k in labels.keys() else 0}%)')
pprint((failure_modes))