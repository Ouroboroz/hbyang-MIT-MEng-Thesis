from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='iwildcam', help='Dataset name')
parser.add_argument('--model', type=str, default="ERM", help="Model name to featurize dataset, not all models for a dataset, Default: ERM")
parser.add_argument('--no_rng', action='store_true', help='Toggles fixed RNG generator for consistency, Default: True') 
parser.add_argument('--cutoff', type=int, default=25, help='Cutoff for number of shots to test, Default: 25')
sampling_algorithms_dict = {'balanced':'balanced_sample_ind', 'full':'full_kmeans_sample_ind', 'class':'class_kmeans_sample_ind', 'iterative':'iterative_kmeans_sample_ind', 'weighted':'weighted_iterative_kmeans_sample_ind'}
parser.add_argument('--sampler', type=str, choices=sampling_algorithms_dict.keys(), default='balanced', help='Sampling algorithms to select training data, Default: balanced')

parser.add_argument('--features', type=str, help='Optional path to featurize data of the dataset/model specified')
parser.add_argument('--classifier', type=str, help='Optional path to the original classifier of the dataset/model specified')
parser.add_argument('--results', type=str, default='../results', help='Optional path to store accuracy results')

config = parser.parse_args()

if config.features is None:
    config.features = f'/dccstor/hoo-misha-1/wilds/wilds/features/{config.dataset}/{config.model}'

set_path_base(config.features)

if config.classifier is None:
    config.classifier = f'/dccstor/hoo-misha-1/wilds/WOODS/classifiers/{config.dataset}/{config.dataset}_{config.model}_classifier'

if config.no_rng:
    rng = np.random.default_rng(2022)
    random_state = 0
else:
    rng = np.random.default_rng()
    random_state = None
set_random_state(rng, random_state)

sampling_algorithm_name = sampling_algorithms_dict[config.sampler]
sampling_algorithm = globals()[sampling_algorithm_name]

print(f'Model: {config.model}')
print(f'Features stored at: {config.features}')
print(f'Original classifier stored at: {config.classifier}')
print(f'Results stored at: {config.results}')
print(f'Fixed RNG State: {config.no_rng}')
print(f'Cutoff: {config.cutoff}')

cam_ids = prune_cam_id()
print(f'Total {len(cam_ids)} to check')
cam_dict = {}
ba_cam_dict = {}
f1_cam_dict = {}
orig_dict = {}
ba_orig_dict = {}
f1_orig_dict = {}

for cam_id in cam_ids:
    print(f'| Cam ID {cam_id}')
    cam_dict[cam_id] = []
    ba_cam_dict[cam_id] = []
    f1_cam_dict[cam_id] = []
    orig_dict[cam_id] = get_original_accuracy(config.classifier, cam_id=[cam_id], cutoff=config.cutoff)
    ba_orig_dict[cam_id] = get_balanced_original_accuracy(config.classifier, cam_id=[cam_id], cutoff=config.cutoff)
    f1_orig_dict[cam_id] = get_original_f1_macro(config.classifier, cam_id=[cam_id], cutoff=config.cutoff)
    print(f'|   | Original Accuracy: O {orig_dict[cam_id]}, BA {ba_orig_dict[cam_id]}, F1 {f1_orig_dict[cam_id]}')
    f,l,m = cam_flm(1, [cam_id])
    f,l,m = prune_flm(f,l,m, config.cutoff)
    unique_classes = np.unique(l)
    num_classes = len(unique_classes)
    for num_shots in range(1,config.cutoff):
        sampled_ind = sampling_algorithm(f,l,num_shots=num_shots)
        nonsampled_ind = np.ones(l.shape[0]) == 1
        nonsampled_ind[sampled_ind] = False
        print(f'|   | Number of Shots:{num_shots}')
        prediction_acc = 0
        balanced_acc = 0
        f1_acc = 0
        c_a = 0
        c_b = 0
        c_f = 0
        for i in range(3):
            prediction_acc_i = get_prediction_accuracy(sampled_ind, nonsampled_ind,cam_id = [cam_id], cutoff=config.cutoff)#, num_shots=num_shots)
            balanced_acc_i = get_balanced_accuracy(sampled_ind, nonsampled_ind,cam_id = [cam_id], cutoff=config.cutoff)
            f1_acc_i = get_f1_macro(sampled_ind, nonsampled_ind,cam_id = [cam_id], cutoff=config.cutoff)
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
            
        print(f'|   |   | Prediction Accuracy: O {prediction_acc}, BA {balanced_acc}, F1 {f1_acc}')
        cam_dict[cam_id].append(prediction_acc)
        ba_cam_dict[cam_id].append(balanced_acc)
        f1_cam_dict[cam_id].append(f1_acc)
    
import pickle

with open(f'{config.results}/{config.model}_cam_dict.pkl','wb') as file:
    pickle.dump(cam_dict, file)
    
with open(f'{config.results}/{config.model}_orig_dict.pkl','wb') as file:
    pickle.dump(orig_dict, file)
    
with open(f'{config.results}/{config.model}_ba_cam_dict.pkl','wb') as file:
    pickle.dump(ba_cam_dict, file)
    
with open(f'{config.results}/{config.model}_ba_orig_dict.pkl','wb') as file:
    pickle.dump(ba_orig_dict, file)
    
with open(f'{config.results}/{config.model}_f1_cam_dict.pkl','wb') as file:
    pickle.dump(f1_cam_dict, file)
    
with open(f'{config.results}/{config.model}_f1_orig_dict.pkl','wb') as file:
    pickle.dump(f1_orig_dict, file)
