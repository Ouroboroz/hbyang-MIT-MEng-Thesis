from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ERM")
parser.add_argument('--features', type=str)

model = 'ERM'
path_base = f'/dccstor/hoo-misha-1/wilds/wilds/features/iwildcam/{model}'
set_path_base(path_base)
rng = np.random.default_rng(2022)
random_state = 0
set_random_state(rng, random_state)

cam_ids = prune_cam_id()
print(f'Total {len(cam_ids)} to check')
cam_dict = {}
ba_cam_dict = {}
f1_cam_dict = {}
orig_dict = {}
ba_orig_dict = {}
f1_orig_dict = {}

cutoff = 25
for cam_id in cam_ids:
    print(f'| Cam ID {cam_id}')
    cam_dict[cam_id] = []
    ba_cam_dict[cam_id] = []
    f1_cam_dict[cam_id] = []
    orig_dict[cam_id] = get_original_accuracy(cam_id=[cam_id], cutoff=cutoff)
    ba_orig_dict[cam_id] = get_balanced_original_accuracy(cam_id=[cam_id], cutoff=cutoff)
    f1_orig_dict[cam_id] = get_original_f1_macro(cam_id=[cam_id], cutoff=cutoff)
    print(f'|   | {orig_dict[cam_id]}')
    f,l,m = cam_flm(1, [cam_id])
    f,l,m = prune_flm(f,l,m, cutoff)
    # sampled_ind_full = kmeans_closest_classes_sample_ind(f,l,batch=cutoff)
    unique_classes = np.unique(l)
    num_classes = len(unique_classes)
    for batch in range(1,cutoff):
        sampled_ind = balanced_sample_ind(l,batch=batch)
        #sampled_ind = kmeans_argmax_n_classes_sample_ind(f,l,batch=batch)
        print(sampled_ind)
        # sampled_ind = sampled_ind_full[:batch*num_classes]
        nonsampled_ind = np.ones(l.shape[0]) == 1
        nonsampled_ind[sampled_ind] = False
        print(f'|   | {batch}')
        prediction_acc = 0
        balanced_acc = 0
        f1_acc = 0
        c_a = 0
        c_b = 0
        c_f = 0
        for i in range(3):
            # sampled_ind = kmeans_argmax_n_classes_sample_ind(f,l,batch=batch)
            # # sampled_ind = sampled_ind_full[:batch*num_classes]
            # nonsampled_ind = np.ones(l.shape[0]) == 1
            # nonsampled_ind[sampled_ind] = False
            prediction_acc_i = get_prediction_accuracy(sampled_ind, nonsampled_ind,cam_id = [cam_id], cutoff=cutoff)#, batch=batch)
            balanced_acc_i = get_balanced_accuracy(sampled_ind, nonsampled_ind,cam_id = [cam_id], cutoff=cutoff)
            f1_acc_i = get_f1_macro(sampled_ind, nonsampled_ind,cam_id = [cam_id], cutoff=cutoff)
            if prediction_acc_i >= 0:
                prediction_acc += prediction_acc_i
                c_a += 1
            if balanced_acc_i >= 0:
                balanced_acc += balanced_acc_i
                c_b += 1
            if f1_acc >= 0:
                f1_acc += f1_acc_i
                c_f += 1
                
        if c_a != 0:
            prediction_acc /= c_a
        else:
            prediction_acc = -1
            
        if c_b != 0:
            balanced_acc /= c_b
        else:
            balanced_acc = -1
        
        if c_f != 0:
            f1_acc /= c_f
        else:
            f1_acc = -1
            
        print(f'|   | {prediction_acc}')
        cam_dict[cam_id].append(prediction_acc)
        ba_cam_dict[cam_id].append(balanced_acc)
        f1_cam_dict[cam_id].append(f1_acc)
import pickle

with open(f'../results/{model}_cam_dict.pkl','wb') as file:
    pickle.dump(cam_dict, file)
    
with open(f'../results/{model}_orig_dict.pkl','wb') as file:
    pickle.dump(orig_dict, file)
    
with open(f'../results/{model}_ba_cam_dict.pkl','wb') as file:
    pickle.dump(ba_cam_dict, file)
    
with open(f'../results/{model}_ba_orig_dict.pkl','wb') as file:
    pickle.dump(ba_orig_dict, file)
    
with open(f'../results/{model}_f1_cam_dict.pkl','wb') as file:
    pickle.dump(f1_cam_dict, file)
    
with open(f'../results/{model}_f1_orig_dict.pkl','wb') as file:
    pickle.dump(f1_orig_dict, file)