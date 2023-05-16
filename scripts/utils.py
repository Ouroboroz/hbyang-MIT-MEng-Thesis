import os

import numpy as np
import pandas as pd
import faiss
from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.nn import Linear

import scipy.stats

import copy

def set_path_base(path):
    global path_base
    path_base = path
    
def set_random_state(r, rs):
    global random_state, rng
    rng = r
    random_state = rs

def set_model_name(mn):
    global model_name
    model_name = mn

def set_config(cf):
    global config
    config = cf 

def load_flm():
    global path_base, model_name
    test_features = np.load(f'{path_base}/{model_name}_test_features.npy')
    test_labels = np.load(f'{path_base}/{model_name}_test_labels.npy')
    test_metadata = np.load(f'{path_base}/{model_name}_test_metadata.npy')
    return test_features, test_labels, test_metadata

def prune_cam_id(cutoff=50):
    global path_base, model_name
    metadata = np.load(f'{path_base}/{model_name}_test_metadata.npy')
    unique_counts = np.unique(metadata[:,0],return_counts=True)
    return unique_counts[0][unique_counts[1] > cutoff]

def get_cam_ind(metadata, num_cams=1, cam_id = None):
    unique_counts = np.unique(metadata[:,0],return_counts=True)
    if cam_id is None:
        top_id = unique_counts[0][np.argpartition(unique_counts[1], -num_cams)[-num_cams:]]
    else:
        top_id = cam_id
    # print(f'Selecting cameras with ids {top_id}')
    ind = np.zeros(metadata.shape[0]) == 1
    for c_id in top_id:
        ind = np.logical_or(ind,metadata[:,0] == c_id)
    return ind

def cam_flm(num_cams=1, cam_id = None):
    features, labels, metadata = load_flm()
    cam_ind = get_cam_ind(metadata, num_cams, cam_id)
    return features[cam_ind], labels[cam_ind], metadata[cam_ind]

def prune_flm(features, labels, metadata, cutoff=25):
    unique_counts = np.unique(labels,return_counts=True)
    #print(f'|   | Total number of classes {len(unique_counts[0])}')
    prune_classes = unique_counts[0][unique_counts[1] < cutoff]
    prune_ind = []
    for clss in prune_classes:
        prune_ind.append((labels == clss).nonzero()[0])
    #print(f'|   |   | Pruning {len(prune_classes)} classes with {len(np.concatenate(prune_ind))} data points')
    if len(prune_ind) == 0:
        return features, labels, metadata
    prune_ind = np.concatenate(prune_ind)
    pruned_ind = np.ones(labels.shape[0]) == 1
    pruned_ind[prune_ind] = False
    return features[pruned_ind], labels[pruned_ind], metadata[pruned_ind]

def balanced_sample_ind(features, labels, num_shots = 5):
    unique_classes = np.unique(labels)
    #print(unique_classes)
    ret_ind = None
    for clss in unique_classes:
        class_ind = np.where(labels == clss)[0]
        #print(clss, class_ind)
        rand_ind = rng.choice(class_ind,num_shots)
        if ret_ind is None:
            ret_ind = rand_ind
        else:
            ret_ind = np.concatenate((ret_ind, rand_ind))
    return ret_ind

def e_dist(A, B, cosine=False, eps=1e-10):
    
    A_n = (A**2).sum(axis=1).reshape(-1,1)
    B_n = (B**2).sum(axis=1).reshape(1,-1)
    inner = A @ B.T
    if cosine:
        return 1 - inner/(np.sqrt(A_n*B_n) + eps)
    else:
        return np.sqrt(np.maximum(0., A_n - 2*inner + B_n))

def euclidean_dist(distances):
    return distances

def gaussian_dist(distances):
    gaussian = scipy.stats.norm(0, 1)
    return gaussian.pdf(distances)


def get_kmeans(features, num_clusters):
    global random_state
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, random_state=random_state)
        km.fit(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, random_state=random_state)
        km.fit(features)
    return km

def full_kmeans_sample_ind(features, labels, num_shots=5, skip_mean=False):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    #kmeans = KMeans(n_clusters=num_shots*num_classes, random_state=random_state).fit(features)
    kmeans = get_kmeans(features, num_clusters=num_shots*num_classes)
    centers = kmeans.cluster_centers_
    cluster_top = {}
    for center_index in range(centers.shape[0]):
        center = centers[center_index]
        for feature_index in range(features.shape[0]):
            feature = features[feature_index]
            dist = np.linalg.norm(feature-center)
            if skip_mean and center == feature:
                    continue
            if center_index not in cluster_top:
                cluster_top[center_index] = (feature_index, dist)
            else:
                if cluster_top[center_index][1] < dist:
                    cluster_top[center_index] = (feature_index, dist)
    ret_ind = []
    for center_index, tup in cluster_top.items():
        ret_ind.append(tup[0])
    return np.array(ret_ind)

def class_kmeans_sample_ind(features, labels, num_shots=5):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    #kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(features)
    kmeans = get_kmeans(features, num_clusters=num_classes)
    centers = kmeans.cluster_centers_
    cluster_top = {}
    for center_index in range(centers.shape[0]):
        center = centers[center_index]
        for feature_index in range(features.shape[0]):
            feature = features[feature_index]
            dist = np.linalg.norm(feature-center)
            if center_index not in cluster_top:
                cluster_top[center_index] = [(feature_index, dist)]
            else:
                # print(cluster_top[center_index])
                cluster_top[center_index].append((feature_index, dist))
    ret_ind = []
    for center_index, tup_list in cluster_top.items():
        tup_list.sort(key = lambda x : x[1])
        for i in range(num_shots):
            ret_ind.append(tup_list[i][0])
       # print(ret_ind)
    return np.array(ret_ind)

def top_n_kmeans_sample_ind(features, labels, num_shots=5, n=100):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    kmeans = KMeans(n_clusters=n*num_classes, random_state=random_state).fit(features)
    centers = kmeans.cluster_centers_
    cluster_top = {}
    for center_index in rng.choice(centers.shape[0], num_shots*num_classes):
        center = centers[center_index]
        for feature_index in range(features.shape[0]):
            feature = features[feature_index]
            dist = np.linalg.norm(feature-center)
            if center_index not in cluster_top:
                cluster_top[center_index] = (feature_index, dist)
            else:
                print(cluster_top[center_index])
                if cluster_top[center_index][1] < dist:
                    cluster_top[center_index] = (feature_index, dist)
    ret_ind = []
    for center_index, tup in cluster_top.items():
        ret_ind.append(tup[0])
    return np.array(ret_ind)

def iterative_kmeans_sample_ind_helper(features, num_classes, num_shots=5, n=100):
    
    if config.precluster:
        if features.shape[0] > n*num_classes:
            num_shot_class_kmeans = get_kmeans(features, num_clusters=n*num_classes)
            num_shot_class_centers = num_shot_class_kmeans.cluster_centers_
        elif features.shape[0]//3 > num_shots * num_classes:
            num_shot_class_kmeans = get_kmeans(features, num_clusters=features.shape[0]//3)
            num_shot_class_centers = num_shot_class_kmeans.cluster_centers_
        else:
            raise Exception('Cannot precluster correctly')
    else:
        num_shot_class_centers = features
    
    kmeans  = get_kmeans(num_shot_class_centers, num_clusters=num_classes)
    centers = copy.copy(kmeans.cluster_centers_)
    if config.typicality:
        center_labels = kmeans.labels_
        center_label_uniques = np.unique(center_labels)
        for label_ind in range(len(center_label_uniques)):
            label = center_label_uniques[label_ind]
            label_indices = center_labels == label 
            label_features = num_shot_class_centers[label_indices]

            center_pairwise_dists = e_dist(label_features, label_features)
            center_pairwise_dists = np.sum(center_pairwise_dists, 1)
            center_min_dist_arg = np.argmin(center_pairwise_dists)

            centers[label_ind] = label_features[center_min_dist_arg]

    
    return num_shot_class_centers, centers

def progressive_iterative_kmeans_sample_ind(config, feature_centers, centers, mask, center_labels = None, center_label_count_dict = None, feature_center_density = None):
    pairwise_distances = e_dist(feature_centers, centers)
    pairwise_distances = config.phi_func(pairwise_distances)
    if config.weighted:
        # center_label_counts = [center_label_count_dict[x]/len(center_labels) for x in center_labels]
        center_label_counts = [1 - center_label_count_dict[x]/len(center_labels) for x in center_labels]
        # print(pairwise_distances.shape, len(center_label_counts))
        distance_sums = pairwise_distances @ np.array(center_label_counts).reshape(-1,1)
    else:
        distance_sums = np.sum(pairwise_distances, 1, keepdims=True)
    distance_sums = distance_sums * mask
    distances_sums = distance_sums / sum(centers)

    if config.density:
        feature_center_density_K = np.partition(feature_center_density, config.K, axis=1)[:,:config.K]
        feature_center_density_K = config.phi_func(feature_center_density_K)
        feature_center_density_K = np.sum(feature_center_density_K, axis=1, keepdims=True)
        distance_sums =  distance_sums - config.lambd * feature_center_density_K / config.K
    
    argm_distance = config.argm(distance_sums)
    #print(argmax_distance)
    mask[argm_distance] = False
    if config.density:
        feature_center_density[:, argm_distance] = np.inf
    centers = np.vstack((centers, feature_centers[argm_distance]))

    return centers, mask


def iterative_kmeans_sample_ind(features, labels, num_shots=5, n=100):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    if features.shape[0] > n*num_classes:
        #num_shots_classes_kmeans = KMeans(n_clusters=n*num_classes, random_state=random_state).fit(features)
        num_shots_classes_kmeans = get_kmeans(features, num_clusters=n*num_classes)
        num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    elif features.shape[0]//3 > num_shots * num_classes:
        #num_shots_classes_kmeans = KMeans(n_clusters=features.shape[0]//3, random_state=random_state).fit(features)
        num_shots_classes_kmeans = get_kmeans(features, num_clusters=features.shape[0]//3)
        num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    else:
        #num_shots_classes_kmeans = KMeans(n_clusters=num_shots*num_classes, random_state=random_state).fit(features)
        num_shots_classes_kmeans = get_kmeans(features, num_clusters=num_shots*num_classes)
        num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    # else:
    #     num_shots_classes_kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(features)
    #     num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    
    kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(num_shots_classes_centers)
    centers = copy.copy(kmeans.cluster_centers_)
    mean_dist = {}
    
    def get_dist(mu_m, mu_is):
        sum_d = 0
        for mu_i in mu_is:
            sum_d += np.linalg.norm(mu_m - mu_i)
        return sum_d
    
    shots_count = 0
    while(shots_count < num_shots*num_classes):
        max_dist = -1
        max_dist_i = None
        for mu_m_i in range(len(num_shots_classes_centers)):
            mu_m = num_shots_classes_centers[mu_m_i]
            if mu_m in centers:
                continue
            mu_m_dist = get_dist(mu_m, centers)
            if mu_m_dist > max_dist:
                max_dist = mu_m_dist
                max_dist_i = mu_m_i
        
        #print(centers.shape)
        try:
            centers = np.vstack((centers, num_shots_classes_centers[max_dist_i][np.newaxis,:]))
        except:
            pass
            #print(f'MESSED UP CENTERS { num_shots_classes_centers[max_dist_i][np.newaxis,:] }')
        shots_count += 1
        
    ret_ind = []  
    for center_index in range(centers.shape[0]):
        center = centers[center_index]
        min_dist = 9999
        min_i = None
        for feature_index in range(features.shape[0]):
            feature = features[feature_index]
            dist = np.linalg.norm(feature-center)
            if dist < min_dist:
                min_dist = dist
                min_i = feature_index
        ret_ind.append(min_i)
    print(ret_ind)
    return np.array(ret_ind)

def weighted_iterative_kmeans_sample_ind(features, labels, num_shots=5, n=100):
    #start_time = time.perf_counter()
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    #print(features.shape, n*num_classes)
    num_shots_classes_kmeans = KMeans(n_clusters=n*num_classes, random_state=random_state).fit(features)
    #print(f'First kmeans: {time.perf_counter() - start_time}')
    start_time = time.perf_counter()
    num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(num_shots_classes_centers)
    #print(f'Second kmeans: {time.perf_counter() - start_time}')
    #start_time = time.perf_counter()
    centers = copy.copy(kmeans.cluster_centers_)
    #print(f'Copy kmeans centers: {time.perf_counter() - start_time}')
    #start_time = time.perf_counter()
    num_shots_classes_center_labels = [-1]*len(num_shots_classes_centers)
    center_labels = [-1] * len(centers)
    
    for i in range(len(features)):
        feature = features[i]
        try:
            ii = centers.index(feature)
            center_labels[ii] = labels[i]
            
        except:
            pass
        
        try:
            ii = num_shots_classes_centers.index(feature)
            num_shots_classes_center_labels[ii] = labels[i]
        except:
            pass
    #print(f'Generate labels: {time.perf_counter() - start_time}')
    #start_time = time.perf_counter()
    mean_dist = {}
    
    def get_dist(mu_m, mu_is, mu_is_labels):
        sum_d = 0
        
        mu_is_labels_counts = np.unique(mu_is_labels, return_counts=True)
        
        mu_is_labels_count_dict = {}
        for label_i in range(len(mu_is_labels_counts[0])):
            label = mu_is_labels_counts[0][label_i]
            counts = mu_is_labels_counts[1][label_i]
            mu_is_labels_count_dict[label] = counts
        
        for mu_i_i in range(len(mu_is)):
            mu_i = mu_is[mu_i_i]
            mu_i_label = mu_is_labels[mu_i_i]
            
            mu_i_label_count = mu_is_labels_count_dict[mu_i_label]
            sum_d += (1 - mu_i_label_count / len(mu_is)) * np.linalg.norm(mu_m - mu_i)
        return sum_d

    shots_count = 0
    while(shots_count < num_shots*num_classes):
        max_dist = -1
        max_dist_i = None
        
        #center_labels_counts = np.unique(center_labels)
        
        for mu_m_i in range(len(num_shots_classes_centers)):
            
            mu_m = num_shots_classes_centers[mu_m_i]
            #mu_m_label = num_shots_classes_center_labels[mu_m_i]

            #mu_m_label_count_i = center_labels_counts[0].index(mu_m_label)
            #mu_m_label_count = center_labels_counts[1][mu_m_label_count_i]
            
            if mu_m in centers:
                continue
            #mu_m_dist = mu_m_label_count / len(center_labels) * get_dist(mu_m, centers)
            mu_m_dist = get_dist(mu_m, centers, center_labels)
            if mu_m_dist > max_dist:
                max_dist = mu_m_dist
                max_dist_i = mu_m_i
        if max_dist_i is None:
            print(max_dist, mu_m_dist)
        try:
            centers = np.vstack((centers, num_shots_classes_centers[max_dist_i][np.newaxis,:]))
        except:
            print(num_shots_classes_centers[max_dist_i].shape, num_shots_classes_centers[max_dist_i][np.newaxis,:].shape)
            centers = np.vstack((centers, np.squeeze(num_shots_classes_centers[max_dist_i])))
        center_labels.append(num_shots_classes_center_labels[max_dist_i])
        #ret_ind.append(max_dist_i)
        shots_count += 1
    #print(f'Get shots {time.perf_counter() - start_time}')
    #start_time = time.perf_counter()
    ret_ind = []  
    for center_index in range(centers.shape[0]):
        center = centers[center_index]
        min_dist = 9999
        min_i = None
        for feature_index in range(features.shape[0]):
            feature = features[feature_index]
            dist = np.linalg.norm(feature-center)
            if dist < min_dist:
                min_dist = dist
                min_i = feature_index
        ret_ind.append(min_i)
    #print(f'Finish {time.perf_counter() - start_time}')
    return np.array(ret_ind)

def get_prediction_accuracy(sampled_ind, nonsampled_ind, f, l):# num_cams=1, largest=True, cam_id = None, cutoff = 25):#, num_shots = 5):
    global random_state
    # f,l,m = cam_flm(num_cams, cam_id)
    # f,l,m = prune_flm(f,l,m, cutoff)
    # sampled_ind = kmeans_closest_num_shots_classes_sample_ind(f,l,num_shots)
    # nonsampled_ind = np.ones(l.shape[0]) == 1
    # nonsampled_ind[sampled_ind] = False
    try:
        clf = LogisticRegression(random_state=random_state, max_iter=100000).fit(f[sampled_ind], l[sampled_ind])
        predictions = clf.predict(f[nonsampled_ind])
    except Exception as e:
        print(f'Count not solve regression {e}')
        return -1
    
    return np.sum(predictions == l[nonsampled_ind])/len(predictions)

def get_detection_prediction_accuracy(sampled_ind, nonsampled_ind, f, l, postprocess):# num_cams=1, largest=True, cam_id = None, cutoff = 25):#, num_shots = 5):
    global random_state
    # f,l,m = cam_flm(num_cams, cam_id)
    # f,l,m = prune_flm(f,l,m, cutoff)
    # sampled_ind = kmeans_closest_num_shots_classes_sample_ind(f,l,num_shots)
    # nonsampled_ind = np.ones(l.shape[0]) == 1
    # nonsampled_ind[sampled_ind] = False
    cls_score_predictor = LogisticRegression(random_state=random_state)
    bbox_pred_predictor = LogisticRegression(random_state=random_state)
    try:
        cls_score = cls_score_predictor.fit()
        predictions = clf.predict(f[nonsampled_ind])
    except Exception as e:
        print(f'Count not solve regression {e}')
        return -1
    
    return np.sum(predictions == l[nonsampled_ind])/len(predictions)

def get_balanced_accuracy(sampled_ind, nonsampled_ind, f, l):# num_cams=1, largest=True, cam_id = None, cutoff = 25, num_shots = 5):
    # f,l,m = cam_flm(num_cams, cam_id)
    # f,l,m = prune_flm(f,l,m, cutoff)
    # sampled_ind = balanced_sample_ind(l,num_shots)
    # nonsampled_ind = np.ones(l.shape[0]) == 1
    # nonsampled_ind[sampled_ind] = False
    try:
        clf = LogisticRegression(random_state=0,max_iter=2000).fit(f[sampled_ind], l[sampled_ind])
        predictions = clf.predict(f[nonsampled_ind])
    except Exception as e:
        #print(e)
        return -1
    
    return balanced_accuracy_score(l[nonsampled_ind], predictions)

def get_f1_macro(sampled_ind, nonsampled_ind, f, l):#num_cams=1, largest=True, cam_id = None, cutoff = 25, num_shots = 5):
    # f,l,m = cam_flm(num_cams, cam_id)
    # f,l,m = prune_flm(f,l,m, cutoff)
    # sampled_ind = balanced_sample_ind(l,num_shots)
    # nonsampled_ind = np.ones(l.shape[0]) == 1
    # nonsampled_ind[sampled_ind] = False
    try:
        clf = LogisticRegression(random_state=0,max_iter=2000).fit(f[sampled_ind], l[sampled_ind])
        predictions = clf.predict(f[nonsampled_ind])
    except Exception as e:
        #print(e)
        return -1
    
    return f1_score(l[nonsampled_ind], predictions,average='macro')


original_weight = None
original_bias = None

def set_original_classifier(path=None):
    global original_weight, original_bias
    original_weight = np.load(f'{path}_weight.npy')
    original_bias = np.load(f'{path}_bias.npy')
    
def get_original_accuracy(f, l):#, num_cams=1, largest=True, cam_id = None, cutoff = 25):
    global original_weight, original_bias
    #f,l,m = cam_flm(num_cams, cam_id)
    #f,l,m = prune_flm(f,l,m, cutoff)
    pred_logits = f @ original_weight.T + original_bias
    pred = np.argmax(pred_logits,axis=1)
    return np.sum(pred == l)/len(pred)

def get_balanced_original_accuracy(f, l): #num_cams=1, largest=True, cam_id = None, cutoff = 25):
    global original_weight, original_bias
    #f,l,m = cam_flm(num_cams, cam_id)
    #f,l,m = prune_flm(f,l,m, cutoff)
    pred_logits = f @ original_weight.T + original_bias
    pred = np.argmax(pred_logits,axis=1)
    return balanced_accuracy_score(l, pred)

def get_original_f1_macro(f, l):#num_cams=1, largest=True, cam_id = None, cutoff = 25):
    global original_weight, original_bias
    #f,l,m = cam_flm(num_cams, cam_id)
    #f,l,m = prune_flm(f,l,m, cutoff)
    pred_logits = f @ original_weight.T + original_bias
    pred = np.argmax(pred_logits,axis=1)
    return f1_score(l, pred,average='macro')

def get_nn(features, num_neighbors):
    # print(features.shape)
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    # print(cpu_index.d)
    gpu_index = cpu_index
    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    # print(gpu_index.d)
    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]

def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_

class TypiClust:
    MIN_CLUSTER_SIZE = 0
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, budgetSize, is_scan=False, f=None, ds_name='iwildcam', model='ERM'):
        self.seed = np.random.randint(100)
        self.lSet = []
        self.features = None
        self.clusters = None
        self.budgetSize = budgetSize
        
        if f is None:
            self.ds_name = ds_name
            self.model = model
            self.init_features_and_clusters(is_scan)
        else:
            self.init_features(f)
        
    def init_features(self, f):
        num_clusters = min(len(self.lSet) + self.budgetSize, self.MAX_NUM_CLUSTERS)
        self.features = f
        self.clusters = kmeans(self.features, num_clusters=num_clusters)
        
    def init_features_and_clusters(self, is_scan):
        num_clusters = min(len(self.lSet) + self.budgetSize, self.MAX_NUM_CLUSTERS)
        # num_clusters = self.MAX_NUM_CLUSTERS
        print(f'Clustering into {num_clusters} clustering. Scan clustering: {is_scan}')
        if is_scan:
            fname_dict = {'CIFAR10': f'../../scan/results/cifar-10/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'CIFAR100': f'../../scan/results/cifar-100/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'TINYIMAGENET': f'../../scan/results/tiny-imagenet/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'iwildcam': f'/dccstor/hoo-misha-1/wilds/wilds/features/iwildcam/{self.model}/resnet50_test_features.npy'
                          }
            fname = fname_dict[self.ds_name]
            self.features = np.load(fname)
            self.clusters = np.load(fname.replace('features', 'probs')).argmax(axis=-1)
        else:
            fname_dict = {'CIFAR10': f'../../scan/results/cifar-10/pretext/features_seed{self.seed}.npy',
                          'CIFAR100': f'../../scan/results/cifar-100/pretext/features_seed{self.seed}.npy',
                          'TINYIMAGENET': f'../../scan/results/tiny-imagenet/pretext/features_seed{self.seed}.npy',
                          'iwildcam': f'/dccstor/hoo-misha-1/wilds/wilds/features/iwildcam/{self.model}/resnet50_test_features.npy',
                          'IMAGENET50': '../../../dino/runs/trainfeat.pth',
                          'IMAGENET100': '../../../dino/runs/trainfeat.pth',
                          'IMAGENET200': '../../../dino/runs/trainfeat.pth',
                          }
            fname = fname_dict[self.ds_name]
            self.features = np.load(fname)
            self.clusters = kmeans(self.features, num_clusters=num_clusters)
        print(f'Finished clustering into {num_clusters} clusters.')

    def select_samples(self):
        # using only labeled+unlabeled indices, without validation set.
        relevant_indices = np.array(range(len(self.features))).astype(int)
        features = self.features #[relevant_indices]
        labels = np.copy(self.clusters) #[relevant_indicies])
        existing_indices = np.arange(len(self.lSet))
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        if sum(clusters_df.cluster_size > self.MIN_CLUSTER_SIZE) == 0:
            print(clusters_df)
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []

        for i in range(self.budgetSize):
            try:
                cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            except:
                print(labels.shape, self.features.shape, cluster_ids.shape, clusters_df.shape, cluster_labeled_counts.shape, labels[existing_indices].shape)
                return None
            indices = (labels == cluster).nonzero()[0]
            rel_feats = features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            try:
                typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
                idx = indices[typicality.argmax()]
                selected.append(idx)
                labels[idx] = -1
            except:
                #print(indices.shape, typicality.shape)
                idx = indicies[0]
                selected.append(idx)
                labels[idx] = -1
                #assert False

        selected = np.array(selected)
        #assert len(selected) == self.budgetSize, 'added a different number of samples'
        #assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        activeSet = relevant_indices[selected]
        remainSet = np.array(sorted(list(set(relevant_indices) - set(activeSet))))

        #print(f'Finished the selection of {len(activeSet)} samples.')
        #print(f'Active set is {activeSet}')
        return activeSet#, remainSet
    
def typiclust_sample_ind(features, labels, num_shots=5, n=100):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    typiclust = TypiClust(num_classes*num_shots, f = features)
    return typiclust.select_samples()