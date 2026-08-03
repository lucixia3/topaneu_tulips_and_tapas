import os, json, pathlib as pl, random, seaborn as sns, matplotlib.pyplot as plt, numpy as np, copy, shutil, pandas as pd
from tqdm import tqdm
from pprint import pprint
import SimpleITK as sitk

def laterality_invariant_mapping(locs):
    mapping = {
        "1": "1",
        "2": "1",
        "3": "2",
        "4": "2",
        "5": "3",
        "6": "3",
        "7": "4",
        "8": "5",
        "9": "6",
        "10": "6",
        "11": "7",
        "12": "7",
        "13": "8",
        "14": "8",
        "15": "9",
        "16": "9",
        "17": "10",
        "18": "11",
        "19": "11",
        "20": "12",
        "21": "12",
        "22": "13",
        "23": "13",
        "24": "14",
        "25": "14",
        "26": "15",
        "27": "15",
        "28": "16",
        "29": "16",
        "30": "17",
        "31": "17",
        "32": "18",
        "33": "18",
        "34": "19",
        "35": "19",
        "36": "20",
        "37": "21",
        "38": "21",
        "39": "22",
        "40": "22",
        "41": "23",
        "42": "23",
        "43": "24",
        "44": "24",
        "45": "25",
        "46": "25",
        "47": "26",
        "48": "26",
        "49": "27",
        "50": "27",
        "51": "28",
        "52": "28"
    }
    output = [mapping[str(lbl)] for lbl in locs]
    return output

def load_samples(path: pl.Path, disregard_laterality=True, tbp=None):
    samples = {}
    tbp = [f for f in os.listdir(path) if f.endswith('.nii.gz')] if tbp is None else tbp
    for file in tbp:
        with open(path/file, "r") as f:
            cur_samp = json.load(f)
            if disregard_laterality:
                cur_samp['locations'] = laterality_invariant_mapping(cur_samp['locations'])
        samples[file.replace('.json', "")]= cur_samp
    return samples

def load_samples_from_msk(path: pl.Path, disregard_laterality=True, tbp=None):
    samples = {}
    tbp = [f for f in os.listdir(path) if f.endswith('.nii.gz')] if tbp is None else tbp
    for file in tbp:
        msk = sitk.ReadImage(path/file)
        msk = sitk.GetArrayFromImage(msk)
        lbls = [lbl for lbl in np.unique(msk).tolist() if lbl!=0]
        cur_samp = {"locations": lbls}
        if disregard_laterality:
            cur_samp['locations'] = laterality_invariant_mapping(cur_samp['locations'])
        samples[file.replace('.json', "")]= cur_samp
    return samples

def get_stats(samples, disregard_lat=True):
    lim=29 if disregard_lat else 53
    occurances = {i:0 for i in range(1, lim)}
    n = 0
    for name, s in samples.items():
        for lbl in s['locations']:
            occurances[int(lbl)]+=1
            n += 1
    return {k: v/n for k, v in occurances.items()}, occurances, n

def get_ref_stats(_, disregard_lat=True):
    with open("abs_distribution.json", "r") as f:
        absolute = {int(k):v for k, v in json.load(f).items()}
    n = sum(absolute.values())
    relative = {}
    for name in absolute.keys():
        relative[name]=absolute[name]/n        
    return relative, absolute, n

def make_plot(data, fn):
    plot = sns.barplot(data)
    plot.tick_params(axis='x', labelrotation=90)
    plot.set_yticks([0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200])
    #plot.set_yticks(list(range(0, 25)))
    plt.savefig(fn)
    plt.close()
    plt.clf()
    
def check_split(assignments, ref, tol, disregard_laterality, samples):
    assignments = [get_stats({s:samples[s] for s in f}, disregard_laterality)[0] for i, f in enumerate(assignments)]
    present = []
    for k, v in samples.items():
        for cls in v["locations"]:
            if cls not in present: present.append(int(cls))
    discrepancies = [{},{}]
    checks = []
    for var in ref.keys():
        if int(var) not in present: continue # skip classes that are not present in the japanese data for evaluation
        disc_a = assignments[0][var] - ref[var]
        disc_b = assignments[1][var] - ref[var]
        discrepancies[0][var] = disc_a
        discrepancies[1][var] = disc_b
        checks.append(abs(disc_a)<=tol and abs(disc_b)<=tol)
    return all(checks), discrepancies

def find_elligible_flips(occ_a, samples_a, occ_b, samples_b, samples):
    elligible_a = []
    for i,k in enumerate(samples_a):
        s_a = samples[k]
        #print(f"Elligible A: {any([occ_a[int(lbl)]<2 for lbl in s_a["locations"]])}, {[occ_a[int(lbl)] for lbl in s_a["locations"]]}")
        if any([occ_a[int(lbl)]<2 for lbl in s_a["locations"]]):
            continue
        else: elligible_a.append(i)
    
    elligible_b = []
    for i,k in enumerate(samples_b):
        s_b = samples[k]
        if any([occ_b[int(lbl)]<2 for lbl in s_b["locations"]]):
            continue
        else: elligible_b.append(i)
        
    return elligible_a, elligible_b

def find_best_flip(elligible, assig, samples, disc, n):
    best_flip_score = np.inf
    best_flip_idx = None
    for e in elligible:
        s = samples[assig[e]]
        cur_disc = copy.deepcopy(disc)
        for lbl in s["locations"]:
            cur_disc[int(lbl)] -= 1/n
        cur_score = sum([abs(v) for v in cur_disc.values()])
        if cur_score<best_flip_score:
            best_flip_score = cur_score
            best_flip_idx = e
    #print(f"Best flip will result in score of: {best_flip_score}")
    return best_flip_idx, best_flip_score

def check_convergence(sc_0_hist, sc_1_hist, converged, patience, sc_0, sc_1, tol):
    if abs(sc_0_hist-sc_0)<tol and abs(sc_1_hist-sc_1)<tol:
        if patience == 2: return sc_0_hist, sc_1_hist, True, None
        else: return sc_0, sc_1, True, patience+1
    else: return sc_0, sc_1, False, 0

def do_split(samples, folds="0.1-0.9", random_seed=42, tol=0.01, max_iter = 265, disregard_laterality=True):
    # assumes all inputs are json objects in the required format!
    folds = folds.split("-")
    assert len(folds) == 2, f"Can only split for two folds!"
    folds = [float(f) for f in folds]
    assert sum(folds) == 1, f"The sum of all folds must equal 1, currently equals {sum(folds)}!"
    
    ## get init stats
    init_dist, init_abs, _ = get_stats(samples, disregard_laterality)
    make_plot(init_dist, 'init_dist.png')
    
    ## init split storage
    fold_assignments = [None, None]
    for i in range(len(folds)):
        fold_assignments[i] = []
    
    ## init split assignments
    fold_ids = [0, 1]
    cls_in_folds = {k: [] for k,v in enumerate(fold_assignments)}
    if random_seed is not None: random.seed(random_seed)
    for i, s in samples.items(): # sorted for consistency
        
        ###### Assignment Block
        to_fold = None # initially assign such that at least one sample of each class is present in any fold, and if unique only in training fold.
        for lbl in s['locations']:
            if lbl not in cls_in_folds[0]:
                to_fold = 0
                cls_in_folds[0].append(lbl)
                break
            elif lbl not in cls_in_folds[1]:
                to_fold = 1
                cls_in_folds[1].append(lbl)
                break
        else: 
            #print(f"Sample {i} has no classes that are missing, assigning randomly.")
            to_fold = random.choices(fold_ids, folds, k=1)[0]
        ###### Assignment Block

        fold_assignments[to_fold].append(i)
    
    check, disc = check_split(fold_assignments, init_dist, tol, disregard_laterality, samples) 
    if check:
        print("Initial split is already quite good! Exiting!")
        relative_0, absolute_0, n_0 = get_stats({f:samples[f] for f in fold_assignments[0]}, disregard_laterality)
        relative_1, absolute_1, n_1 = get_stats({f:samples[f] for f in fold_assignments[1]}, disregard_laterality)
        make_plot(absolute_0, 'res_dist_0.png')
        make_plot(absolute_1, 'res_dist_1.png')
        return fold_assignments
    else: 
        sc_0_hist = np.inf
        sc_1_hist = np.inf
        converged = False
        patience = 0
        for i in range(max_iter):
            relative_0, absolute_0, n_0 = get_stats({f:samples[f] for f in fold_assignments[0]}, disregard_laterality)
            relative_1, absolute_1, n_1 = get_stats({f:samples[f] for f in fold_assignments[1]}, disregard_laterality)
            
            # find elligible flips -> only flip samples that are non-unique
            elligible_0, elligible_1 = find_elligible_flips(absolute_0, fold_assignments[0], absolute_1, fold_assignments[1], samples)
            
            # find best flip between distributions
            flip_0, sc_0 = find_best_flip(elligible_0, fold_assignments[0], samples, disc[0], n_0)
            flip_1, sc_1 = find_best_flip(elligible_1, fold_assignments[1], samples, disc[1], n_1)
            sc_0_hist, sc_1_hist, converged, patience = check_convergence(sc_0_hist, sc_1_hist, converged, patience, sc_0, sc_1, tol)
            
            # execute, check, repeat
            flip_0 = fold_assignments[0].pop(flip_0)
            flip_1 = fold_assignments[1].pop(flip_1)
            fold_assignments[0].append(flip_1)
            fold_assignments[1].append(flip_0)
            check, disc = check_split(fold_assignments, init_dist, tol, disregard_laterality, samples) 
            if check:
                print("Found good split!")
                make_plot(relative_0, 'res_dist_0.png')
                make_plot(relative_1, 'res_dist_1.png')
                return fold_assignments
            elif converged:
                print("Split converged!")
                make_plot(relative_0, 'res_dist_0.png')
                make_plot(relative_1, 'res_dist_1.png')
                return fold_assignments
            
        else:
            make_plot(relative_0, 'latest_dist_0.png')
            make_plot(relative_1, 'latest_dist_1.png')
            raise RuntimeError(f"Failed to find a split within tolerance {tol} after {max_iter} iterations!")
        
def save_fold_assignments(assignments, paths, src_json, src_images, src_masks, src_types):
    for i, (ass, path) in enumerate(zip(assignments, paths)):
        os.makedirs(path/"location_jsons", exist_ok=True)
        os.makedirs(path/"images", exist_ok=True)
        os.makedirs(path/"type_masks", exist_ok=True)
        os.makedirs(path/"location_masks", exist_ok=True)
        for samp in tqdm(ass, desc=f"Copying Files for fold {i}"):
            fn = f"{samp.removesuffix(".nii.gz")}_0000"
            shutil.copy(src_json/f"{fn.removesuffix("_0000")}.json", path/"location_jsons"/f"{fn.removesuffix("_0000")}.json")
            shutil.copy(src_images/f"{fn}.nii.gz", path/"images"/f"{fn}.nii.gz")
            shutil.copy(src_masks/f"{fn.removesuffix("_0000")}.nii.gz", path/"location_masks"/f"{fn.removesuffix("_0000")}.nii.gz")
            shutil.copy(src_types/f"{fn.removesuffix("_0000")}.nii.gz", path/"type_masks"/f"{fn.removesuffix("_0000")}.nii.gz")

if __name__ == "__main__":
    DISREGARD_LATERALITY = True
    ## In case you have json labels
    # samples = load_samples(pl.Path("/mnt/c/Users/20260926/OneDrive - TU Eindhoven/Documents/Datasets/TopAneu-raw/HUG/TopAneu_center2_batch1/labels"), disregard_laterality=DISREGARD_LATERALITY)
    ## In case you have location masks in .nii.gz
    with open('split.json', 'r') as file:
        tbp = json.load(file)
    
    samples = load_samples_from_msk(pl.Path("/home/tue20260926/Repos/TopAneu-26/topaneu_release/location_masks"), tbp=tbp, disregard_laterality=DISREGARD_LATERALITY)
    
    
    assignments = do_split(samples, disregard_laterality=DISREGARD_LATERALITY, max_iter=1000, tol=0.025)
    
    with open('split.json', 'w') as file:
        json.dump(assignments, file, indent=4)

