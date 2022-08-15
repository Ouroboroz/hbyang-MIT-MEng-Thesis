from utils import *

path_base = '/dccstor/hoo-misha-1/wilds/wilds/features/iwildcam/PseudoLabel'
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
    if f.shape[0] < 500:
        continue
    # sampled_ind_full = kmeans_closest_classes_sample_ind(f,l,batch=cutoff)
    unique_classes = np.unique(l)
    num_classes = len(unique_classes)
    batch_classes_centers, centers = kmeans_argmax_n_classes_sample_ind_helper(f,l)
    
    batch_classes_center_labels = [-1]*len(batch_classes_centers)
    center_labels = [-1] * len(centers)
    
    for i in range(len(f)):
        feature = f[i]
        try:
            ii = centers.index(feature)
            center_labels[ii] = labels[i]
            
        except:
            pass
        
        try:
            ii = batch_classes_centers.index(feature)
            batch_classes_center_labels[ii] = labels[i]
        except:
            pass
        
    distances = {}
    
    def get_dist(mu_m, mu_is):
        sum_d = 0
        for mu_i in mu_is:
            if (tuple(mu_i), tuple(mu_m)) in distances or (tuple(mu_m), tuple(mu_i)) in distances:
                sum_d += distances[(tuple(mu_i), tuple(mu_m))]
            else:
                sum_d += np.linalg.norm(mu_m - mu_i)
                distances[(tuple(mu_i), tuple(mu_m))] = np.linalg.norm(mu_m - mu_i)
                distances[(tuple(mu_m), tuple(mu_i))] = np.linalg.norm(mu_m - mu_i)
        return sum_d
    
    for batch in range(1,cutoff):
        # get sampled ind
        shots_count = 0
        while(shots_count < batch*num_classes):
            max_dist = -1
            max_dist_i = None
            
            center_labels_values, center_labels_counts = np.unique(center_labels, return_counts=True)
            center_labels_values = list(center_labels_values)
            
            for mu_m_i in range(len(batch_classes_centers)):
                mu_m = batch_classes_centers[mu_m_i]
                mu_m_label = batch_classes_center_labels[mu_m_i]

                mu_m_label_count_i = center_labels_values.index(mu_m_label)
                mu_m_label_count = center_labels_counts[mu_m_label_count_i]
                
                if mu_m in centers:
                    continue
                mu_m_dist = (1 - mu_m_label_count / len(center_labels)) * get_dist(mu_m, centers)
                if mu_m_dist > max_dist:
                    max_dist = mu_m_dist
                    max_dist_i = mu_m_i

            #print(centers.shape)
            try:
                centers = np.vstack((centers, batch_classes_centers[max_dist_i][np.newaxis,:]))
            except:
                print(f'MESSED UP CENTERS { batch_classes_centers[max_dist_i][np.newaxis,:] }')
            shots_count += 1

        ret_ind = []  
        for center_index in range(centers.shape[0]):
            center = centers[center_index]
            min_dist = 9999
            min_i = None
            for feature_index in range(f.shape[0]):
                feature = f[feature_index]
                dist = np.linalg.norm(feature-center)
                if dist < min_dist:
                    min_dist = dist
                    min_i = feature_index
            ret_ind.append(min_i)
        sampled_ind = ret_ind
        #sampled_ind = kmeans_argmax_n_classes_sample_ind(f,l,batch=batch)
        #print(sampled_ind)
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

with open('kmeans_argmax_weighted_n_classes_fast_cam_dict.pkl','wb') as file:
    pickle.dump(cam_dict, file)
    
with open('kmeans_argmax_weighted_n_classes_fast_orig_dict.pkl','wb') as file:
    pickle.dump(orig_dict, file)
    
with open('ba_kmeans_argmax_weighted_n_classes_fast_cam_dict.pkl','wb') as file:
    pickle.dump(ba_cam_dict, file)
    
with open('ba_kmeans_argmax_weighted_n_classes_fast_orig_dict.pkl','wb') as file:
    pickle.dump(ba_orig_dict, file)
    
with open('f1_kmeans_argmax_weighted_n_classes_fast_cam_dict.pkl','wb') as file:
    pickle.dump(f1_cam_dict, file)
    
with open('f1_kmeans_argmax_weighted_n_classes_fast_orig_dict.pkl','wb') as file:
    pickle.dump(f1_orig_dict, file)
