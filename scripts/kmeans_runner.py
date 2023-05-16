from utils import *
import argparse 
import pickle
import time
import wandb
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='breeds', help='Dataset name')
parser.add_argument('--model', type=str, default="deepCORAL", help="Model name to featurize dataset, not all models for a dataset, Default: ERM")
parser.add_argument('--model_name', type=str, help="Model name to featurize dataset")
parser.add_argument('--no_rng', action='store_true', help='Toggles fixed RNG generator for consistency, Default: True') 
parser.add_argument('--cutoff', type=int, default=25, help='Cutoff for number of shots to test, Default: 25')

parser.add_argument('--precluster', action='store_true', help='Uses precluster points for initial sample selection')
parser.add_argument('--typicality', action='store_true', help='Uses typicality to select initial sample selection')

parser.add_argument('--sampler', type=str, default='balanced', help='Sampling algorithms to select training data, Default: balanced')
parser.add_argument('--weighted', action='store_true', help='Uses class weights for iterative kmeans')
parser.add_argument('--density', action='store_true', help='Uses additional density information for selecting best point')
phi_func_choices = ['euclidean', 'gaussian']
parser.add_argument('--phi_func', type=str, choices=phi_func_choices, default='euclidean')
parser.add_argument('--lambd', type=float, default=1)
parser.add_argument('--K', type=int, default=20)

parser.add_argument('--use_wandb', action='store_true', help='Uses wandb to track runs')
parser.add_argument('--reset', action='store_true', help='Resets dictionaries to store info')
parser.add_argument('--clean', action='store_true', help='Uses a clean dictionary to run')

parser.add_argument('--features', type=str, help='Optional path to featurize data of the dataset/model specified')
parser.add_argument('--classifier', type=str, help='Optional path to the original classifier of the dataset/model specified')
parser.add_argument('--results', type=str, default='/dccstor/hoo-misha-1/wilds/WOODS/results', help='Optional path to store accuracy results')

config = parser.parse_args()

if config.model_name is None:
    if config.dataset == 'iwildcam':
        config.model_name = 'resnet50'
    elif config.dataset == 'camelyon17':
        config.model_name = 'densenet121'
    elif config.dataset == 'breeds':
        config.model_name = 'resnet50'
    elif config.dataset == 'cifar100':
        config.model_name = 'resnet50'

set_model_name(config.model_name)

if config.features is None:
    config.features = f'/dccstor/hoo-misha-1/wilds/wilds/features/{config.dataset}/{config.model}'

set_path_base(config.features)

if config.classifier is None:
    config.classifier = f'/dccstor/hoo-misha-1/wilds/WOODS/classifiers/{config.dataset}/{config.dataset}_{config.model}_classifier'

if config.no_rng:
    rng = np.random.default_rng(2022)
    config.random_state = 0
else:
    rng = np.random.default_rng()
    config.random_state = None
set_random_state(rng, config.random_state)

set_original_classifier(config.classifier)

# sampling_algorithm_name = sampling_algorithms_dict[config.sampler]
# sampling_algorithm = globals()[sampling_algorithm_name]

set_config(config)

if config.use_wandb:
    wandb.init(project='iterative_breeds')
    wandb.config.update(config)

save_file_path = f'{config.results}/{config.dataset}/{config.model}/{config.model}_iterative_pc:{config.precluster}_typ:{config.typicality}_w:{config.weighted}_d:{config.density}_phi:{config.phi_func}_lambda:{config.lambd}_rng:{config.random_state}'
centers_save_file_path = f'{config.results}/{config.dataset}/{config.model}/{config.model}_iterative_pc:{config.precluster}_typ:{config.typicality}_rng:{config.random_state}'

print(f'Model: {config.model}')
print(f'Features stored at: {config.features}')
print(f'Original classifier stored at: {config.classifier}')
print(f'Results stored at: {save_file_path}')
print(f'Fixed RNG State: {config.no_rng}')
print(f'Cutoff: {config.cutoff}')
print(f'Wandb: {config.use_wandb}')

print(f'Sampler: Iterative')
print(f'Preclustered: {config.precluster}')
print(f'Typicality: {config.typicality}')
print(f'Weighted: {config.weighted}')
print(f'Density: {config.density}')
print(f'Phi Function: {config.phi_func}')
print(f'Lambda: {config.lambd}')

if config.phi_func == 'euclidean':
    config.phi_func = euclidean_dist
    config.argm = np.argmax
elif config.phi_func == 'gaussian':
    config.phi_func = gaussian_dist
    config.argm = np.argmin
else:
    raise Exception('Unknown function')

cam_ids = prune_cam_id()
print(f'Total {len(cam_ids)} to check')

def pickle_dicts():
    with open(f'{save_file_path}_cam_dict.pkl','wb') as file:
        pickle.dump(cam_dict, file)

    with open(f'{save_file_path}_orig_dict.pkl','wb') as file:
        pickle.dump(orig_dict, file)

    with open(f'{save_file_path}_ba_cam_dict.pkl','wb') as file:
        pickle.dump(ba_cam_dict, file)

    with open(f'{save_file_path}_ba_orig_dict.pkl','wb') as file:
        pickle.dump(ba_orig_dict, file)

    with open(f'{save_file_path}_f1_cam_dict.pkl','wb') as file:
        pickle.dump(f1_cam_dict, file)

    with open(f'{save_file_path}_f1_orig_dict.pkl','wb') as file:
        pickle.dump(f1_orig_dict, file)
    
try:
    if config.clean or config.reset:
        raise Exception('Reset')
        
    with open(f'{save_file_path}_cam_dict.pkl','rb') as file:
        cam_dict = pickle.load(file)

    with open(f'{save_file_path}_orig_dict.pkl','rb') as file:
        orig_dict = pickle.load(file)

    with open(f'{save_file_path}_ba_cam_dict.pkl','rb') as file:
        ba_cam_dict = pickle.load(file)

    with open(f'{save_file_path}_ba_orig_dict.pkl','rb') as file:
        ba_orig_dict = pickle.load(file)

    with open(f'{save_file_path}_f1_cam_dict.pkl','rb') as file:
        f1_cam_dict = pickle.load(file)

    with open(f'{save_file_path}_f1_orig_dict.pkl','rb') as file:
        f1_orig_dict = pickle.load(file)
except:
    cam_dict = {}
    ba_cam_dict = {}
    f1_cam_dict = {}
    orig_dict = {}
    ba_orig_dict = {}
    f1_orig_dict = {}
    if config.reset:
        pickle_dicts()


cutoff = config.cutoff
cam_count = 0
for cam_id in cam_ids:
    if cam_id in cam_dict:
        print(f'| Skipping Domain ID {cam_id}')
        continue
    print(f'| Domain ID {cam_id}')
    if cam_count == 5:
        print('Saving dicts')
        pickle_dicts()
        cam_count = 0
    cam_count += 1
    cam_dict[cam_id] = []
    ba_cam_dict[cam_id] = []
    f1_cam_dict[cam_id] = []
    f,l,m = cam_flm(1, [cam_id])
    f,l,m = prune_flm(f,l,m, cutoff)
    
    orig_dict[cam_id] = get_original_accuracy(f, l)#cam_id=[cam_id], cutoff=cutoff)
    ba_orig_dict[cam_id] = get_balanced_original_accuracy(f, l)#cam_id=[cam_id], cutoff=cutoff)
    f1_orig_dict[cam_id] = get_original_f1_macro(f, l)#cam_id=[cam_id], cutoff=cutoff)
    if config.use_wandb:
            wandb.log({'original_accuracy': orig_dict[cam_id], 'original_balanced_accuracy': ba_orig_dict[cam_id], 'original_f1_macro': f1_orig_dict[cam_id]})

    print(f'|   | {orig_dict[cam_id]}')

    unique_classes = np.unique(l)
    num_classes = len(unique_classes)

    try:
        if config.precluster:
            with open(f'{centers_save_file_path}_feature_centers.pkl','rb') as file:
                feature_centers = pickle.load(file)
        else:
            feature_centers = f

        with open(f'{centers_save_file_path}_centers.pkl','rb') as file:
            centers = pickle.load(file)  
    except:
        feature_centers, centers = iterative_kmeans_sample_ind_helper(f,num_classes,num_shots=cutoff)
        print('Saving feature centers')
        if config.precluster:
            with open(f'{centers_save_file_path}_feature_centers.pkl','wb') as file:
                pickle.dump(feature_centers, file)

        with open(f'{centers_save_file_path}_centers.pkl','wb') as file:
            pickle.dump(centers, file)
    
    mask = np.ones((feature_centers.shape[0],1))
    feature_mask = np.zeros((1,f.shape[0]))

    centers_features_pairwise_dist = e_dist(centers, f)
    sampled_ind = np.argmin(centers_features_pairwise_dist,1)
    feature_mask[0,sampled_ind] = np.inf

    if config.weighted:
        center_labels = list(l[sampled_ind])
        center_label_uniques, center_label_counts = np.unique(center_labels, return_counts = True)
        center_label_count_dict = dict(zip(center_label_uniques, center_label_counts))
    else:
        center_labels = None
        center_label_count_dict = None

    if config.density:
        feature_center_density = e_dist(feature_centers, feature_centers)
        # gaussian = scipy.stats.norm(0, 1)
        # feature_center_density = gaussian.pdf(feature_center_density)
        #feature_center_density = np.sum(feature_center_density, 1, keepdims=True)
    else:
        feature_center_density = None

    for batch in range(1,cutoff):
        for i in range(num_classes):
            centers, mask = progressive_iterative_kmeans_sample_ind(config, feature_centers, centers, mask, center_labels = center_labels, center_label_count_dict = center_label_count_dict, feature_center_density = feature_center_density)
            newest_center = centers[[-1]]

            newest_center_features_pairwise_dist = e_dist(newest_center, f) + feature_mask
            newest_ind = np.argmin(newest_center_features_pairwise_dist)

            feature_mask[0,newest_ind] = np.inf
            sampled_ind = np.append(sampled_ind,newest_ind)

            new_center_label = l[newest_ind]
            if center_labels is not None:
                center_labels.append(new_center_label)
                if new_center_label in center_label_count_dict:
                    center_label_count_dict[new_center_label] += 1
                else:
                    center_label_count_dict[new_center_label] = 1

        #sampled_ind = kmeans_argmax_n_classes_sample_ind(f,l,batch=batch)
        #print(len(centers), feature_mask.shape,sampled_ind)
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
            prediction_acc_i = get_prediction_accuracy(sampled_ind, nonsampled_ind, f, l)#,cam_id = [cam_id], cutoff=cutoff)#, batch=batch)
            balanced_acc_i = get_balanced_accuracy(sampled_ind, nonsampled_ind, f, l)#,cam_id = [cam_id], cutoff=cutoff)
            f1_acc_i = get_f1_macro(sampled_ind, nonsampled_ind, f, l)#cam_id = [cam_id], cutoff=cutoff)
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
            continue
            
        if c_b != 0:
            balanced_acc /= c_b
        else:
            balanced_acc = -1
            continue
        
        if c_f != 0:
            f1_acc /= c_f
        else:
            f1_acc = -1
            continue
            
        print(f'|   | {prediction_acc}')
        cam_dict[cam_id].append(prediction_acc)
        ba_cam_dict[cam_id].append(balanced_acc)
        f1_cam_dict[cam_id].append(f1_acc)

        if config.use_wandb:
            wandb.log({f'{cam_id}_accuracy': prediction_acc, f'{cam_id}_balanced_accuracy': balanced_acc, f'{cam_id}_f1_macro': f1_acc})


pickle_dicts()

if config.use_wandb:
    wandb.finish()
