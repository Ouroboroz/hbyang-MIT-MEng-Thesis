import os

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score

import copy

model_name = 'resnet50'

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

def load_flm():
    global path_base, model_name
    test_features = np.load(f'{path_base}/{model_name}_test_features.npy')
    test_labels = np.load(f'{path_base}/{model_name}_test_labels.npy')
    test_metadata = np.load(f'{path_base}/{model_name}_test_metadata.npy')
    return test_features, test_labels, test_metadata

def prune_cam_id(cutoff=50):
    global path_base
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

def full_kmeans_sample_ind(features, labels, num_shots=5, skip_mean=False):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    kmeans = KMeans(n_clusters=num_shots*num_classes, random_state=random_state).fit(features)
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
    kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(features)
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

def iterative_kmeans_sample_ind_helper(features, labels, num_shots=5, n=100):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    if features.shape[0] > n*num_classes:
        num_shots_classes_kmeans = KMeans(n_clusters=n*num_classes, random_state=random_state).fit(features)
        num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    else:
        num_shots_classes_kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(features)
        num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(num_shots_classes_centers)
    centers = copy.copy(kmeans.cluster_centers_)
    
    return num_shots_classes_centers, centers

def iterative_kmeans_sample_ind(features, labels, num_shots=5, n=100):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    if features.shape[0] > n*num_classes:
        num_shots_classes_kmeans = KMeans(n_clusters=n*num_classes, random_state=random_state).fit(features)
        num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    elif features.shape[0]/3 > num_classes:
        um_shots_classes_kmeans = KMeans(n_clusters=features.shape[0]/3, random_state=random_state).fit(features)
        num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
    else:
        num_shots_classes_kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(features)
        num_shots_classes_centers = num_shots_classes_kmeans.cluster_centers_
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

def get_prediction_accuracy(sampled_ind, nonsampled_ind, num_cams=1, largest=True, cam_id = None, cutoff = 25):#, num_shots = 5):
    f,l,m = cam_flm(num_cams, cam_id)
    f,l,m = prune_flm(f,l,m, cutoff)
    # sampled_ind = kmeans_closest_num_shots_classes_sample_ind(f,l,num_shots)
    # nonsampled_ind = np.ones(l.shape[0]) == 1
    # nonsampled_ind[sampled_ind] = False
    try:
        clf = LogisticRegression(random_state=0,max_iter=100000).fit(f[sampled_ind], l[sampled_ind])
        predictions = clf.predict(f[nonsampled_ind])
    except Exception as e:
        print(f'Count not solve regression {e}')
        return -1
    
    return np.sum(predictions == l[nonsampled_ind])/len(predictions)

def get_balanced_accuracy(sampled_ind, nonsampled_ind, num_cams=1, largest=True, cam_id = None, cutoff = 25, num_shots = 5):
    f,l,m = cam_flm(num_cams, cam_id)
    f,l,m = prune_flm(f,l,m, cutoff)
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

def get_f1_macro(sampled_ind, nonsampled_ind, num_cams=1, largest=True, cam_id = None, cutoff = 25, num_shots = 5):
    f,l,m = cam_flm(num_cams, cam_id)
    f,l,m = prune_flm(f,l,m, cutoff)
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


def get_original_accuracy(path=None, num_cams=1, largest=True, cam_id = None, cutoff = 25):
    f,l,m = cam_flm(num_cams, cam_id)
    f,l,m = prune_flm(f,l,m, cutoff)
    weight = np.load(f'{path}_weight.npy')
    bias = np.load(f'{path}_bias.npy')
    pred_logits = f @ weight.T + bias
    pred = np.argmax(pred_logits,axis=1)
    return np.sum(pred == l)/len(pred)

def get_balanced_original_accuracy(path=None, num_cams=1, largest=True, cam_id = None, cutoff = 25):
    f,l,m = cam_flm(num_cams, cam_id)
    f,l,m = prune_flm(f,l,m, cutoff)
    weight = np.load(f'{path}_weight.npy')
    bias = np.load(f'{path}_bias.npy')
    pred_logits = f @ weight.T + bias
    pred = np.argmax(pred_logits,axis=1)
    return balanced_accuracy_score(l, pred)

def get_original_f1_macro(path=None, num_cams=1, largest=True, cam_id = None, cutoff = 25):
    f,l,m = cam_flm(num_cams, cam_id)
    f,l,m = prune_flm(f,l,m, cutoff)
    weight = np.load(f'{path}_weight.npy')
    bias = np.load(f'{path}_bias.npy')
    pred_logits = f @ weight.T + bias
    pred = np.argmax(pred_logits,axis=1)
    return f1_score(l, pred,average='macro')