#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


# In[ ]:


rng = np.random.default_rng(2022)
random_state = 0


# In[ ]:


os.system('ls /dccstor/hoo-misha-1/wilds/wilds/features/iwildcam/')


# In[23]:


path_base = '/dccstor/hoo-misha-1/wilds/wilds/features/iwildcam/PseudoLabel'
os.system('ls /dccstor/hoo-misha-1/wilds/wilds/features/iwildcam/PseudoLabel')


# In[24]:


def load_flm():
    test_features = np.load(f'{path_base}/resnet50_test_features.npy')
    test_labels = np.load(f'{path_base}/resnet50_test_labels.npy')
    test_metadata = np.load(f'{path_base}/resnet50_test_metadata.npy')
    return test_features, test_labels, test_metadata


# In[25]:


def prune_cam_id(cutoff=50):
    metadata = np.load(f'{path_base}/resnet50_test_metadata.npy')
    unique_counts = np.unique(metadata[:,0],return_counts=True)
    return unique_counts[0][unique_counts[1] > cutoff]


# In[26]:


def get_cam_ind(metadata, num_cams=1, cam_id = None):
    unique_counts = np.unique(metadata[:,0],return_counts=True)
    if cam_id is None:
        top_id = unique_counts[0][np.argpartition(unique_counts[1], -num_cams)[-num_cams:]]
    else:
        top_id = cam_id
    print(f'Selecting cameras with ids {top_id}')
    ind = np.zeros(metadata.shape[0]) == 1
    for c_id in top_id:
        ind = np.logical_or(ind,metadata[:,0] == c_id)
    return ind


# In[27]:


def cam_flm(num_cams=1, cam_id = None):
    features, labels, metadata = load_flm()
    cam_ind = get_cam_ind(metadata, num_cams, cam_id)
    return features[cam_ind], labels[cam_ind], metadata[cam_ind]


# In[28]:


def prune_flm(features, labels, metadata, cutoff=25):
    unique_counts = np.unique(labels,return_counts=True)
    print(f'|   | Total number of classes {len(unique_counts[0])}')
    prune_classes = unique_counts[0][unique_counts[1] < cutoff]
    prune_ind = []
    for clss in prune_classes:
        prune_ind.append((labels == clss).nonzero()[0])
    print(f'|   |   | Pruning {len(prune_classes)} classes with {len(np.concatenate(prune_ind))} data points')
    if len(prune_ind) == 0:
        return features, labels, metadata
    prune_ind = np.concatenate(prune_ind)
    pruned_ind = np.ones(labels.shape[0]) == 1
    pruned_ind[prune_ind] = False
    return features[pruned_ind], labels[pruned_ind], metadata[pruned_ind]


# In[29]:


def balanced_sample_ind(labels, batch = 5):
    unique_classes = np.unique(labels)
    #print(unique_classes)
    ret_ind = None
    for clss in unique_classes:
        class_ind = np.where(labels == clss)[0]
        #print(clss, class_ind)
        rand_ind = rng.choice(class_ind,batch)
        if ret_ind is None:
            ret_ind = rand_ind
        else:
            ret_ind = np.concatenate((ret_ind, rand_ind))
    return ret_ind


# In[30]:


def kmeans_closest_batch_classes_sample_ind(features, labels, batch=5, skip_mean=False):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    kmeans = KMeans(n_clusters=batch*num_classes, random_state=random_state).fit(features)
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


# In[98]:


def kmeans_closest_classes_sample_ind(features, labels, batch=5):
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
        for i in range(batch):
            ret_ind.append(tup_list[i][0])
       # print(ret_ind)
    return np.array(ret_ind)


# In[87]:


def kmeans_closest_n_classes_sample_ind(features, labels, batch=5, n=100):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    kmeans = KMeans(n_clusters=n*num_classes, random_state=random_state).fit(features)
    centers = kmeans.cluster_centers_
    cluster_top = {}
    for center_index in rng.choice(centers.shape[0], batch*num_classes):
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


# In[88]:


def kmeans_argmax_n_classes_sample_ind(features, labels, batch=5, n=100):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    if features.shape[0] > n*num_classes:
        batch_classes_kmeans = KMeans(n_clusters=n*num_classes, random_state=random_state).fit(features)
        batch_classes_centers = batch_classes_kmeans.cluster_centers_
    else:
        batch_classes_kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(features)
        batch_classes_centers = batch_classes_kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(batch_classes_centers)
    centers = kmeans.cluster_centers_
    mean_dist = {}
    
    def get_dist(mu_m, mu_is):
        sum_d = 0
        for mu_i in mu_is:
            sum_d += np.linalg.norm(mu_m - mu_i)
        return sum_d
    
    for mu_m_i in range(len(batch_classes_centers)):
        mu_m = batch_classes_centers[mu_m_i]
        if mu_m in centers:
            continue
        mean_dist[mu_m_i] = get_dist(mu_m, batch_classes_centers)
    
    mean_dist_tups = list(mean_dist.items())
    mean_dist_tups.sort(key=lambda x : x[1],reverse=True)
    ret_ind = []
    for i in range(batch*num_classes):
        ret_ind.append(mean_dist_tups[i][0])
    return np.array(ret_ind)


# In[89]:


def kmeans_weighted_argmax_n_classes_sample_ind(features, labels, batch=5, n=100):
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    batch_classes_kmeans = KMeans(n_clusters=n*num_classes, random_state=random_state).fit(features)
    batch_classes_centers = batch_classes_kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=num_classes, random_state=random_state).fit(batch_classes_centers)
    centers = kmeans.cluster_centers_
    
    batch_classes_center_labels = [-1]*len(batch_classes_center_labels)
    center_labels = [-1] * len(centers)
    
    for i in range(len(features)):
        feature = features[i]
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
    
    mean_dist = {}
    
    def get_dist(mu_m, mu_is):
        sum_d = 0
        for mu_i in mu_is:
            sum_d += np.linalg.norm(mu_m - mu_i)
        return sum_d
    
    center_labels_counts = np.unique(center_labels)
    
    for mu_m_i in range(len(batch_classes_centers)):
        mu_m = batch_classes_centers[mu_m_i]
        mu_m_label = batch_classes_center_labels[mu_m_i]
        
        mu_m_label_count_i = center_labels_counts[0].index(mu_m_label)
        mu_m_label_count = center_labels_counts[1][mu_m_label_count_i]
        
        if mu_m in centers:
            continue
        mean_dist[mu_m_i] = mu_m_label_count / len(center_labels) * get_dist(mu_m, batch_classes_kmeans)
    
    mean_dist_tups = list(mean_dist.items())
    mean_dist_tups.sort(key=lambda x : x[1],reverse=True)
    ret_ind = []
    for i in range(batch*num_classes):
        ret_ind.append(mean_dist_tups[i][0])
    return np.array(ret_ind)


# In[90]:


def get_prediction_accuracy(sampled_ind, nonsampled_ind, num_cams=1, largest=True, cam_id = None, cutoff = 25):#, batch = 5):
    f,l,m = cam_flm(num_cams, cam_id)
    f,l,m = prune_flm(f,l,m, cutoff)
    # sampled_ind = kmeans_closest_batch_classes_sample_ind(f,l,batch)
    # nonsampled_ind = np.ones(l.shape[0]) == 1
    # nonsampled_ind[sampled_ind] = False
    try:
        clf = LogisticRegression(random_state=0,max_iter=2000).fit(f[sampled_ind], l[sampled_ind])
        predictions = clf.predict(f[nonsampled_ind])
    except:
        return -1
    
    return np.sum(predictions == l[nonsampled_ind])/len(predictions)


# In[91]:


def get_original_accuracy(num_cams=1, largest=True, cam_id = None, cutoff = 25):
    f,l,m = cam_flm(num_cams, cam_id)
    f,l,m = prune_flm(f,l,m, cutoff)
    weight = np.load('/dccstor/hoo-misha-1/wilds/wilds/pseudo_classifier_weight.npy')
    bias = np.load('/dccstor/hoo-misha-1/wilds/wilds/pseudo_classifier_bias.npy')
    pred_logits = f @ weight.T + bias
    pred = np.argmax(pred_logits,axis=1)
    return np.sum(pred == l)/len(pred)


# In[ ]:


cam_ids = prune_cam_id()
print(f'Total {len(cam_ids)} to check')
cam_dict = {}
orig_dict = {}
cutoff = 25
for cam_id in cam_ids:
    print(f'| Cam ID {cam_id}')
    cam_dict[cam_id] = []
    orig_dict[cam_id] = get_original_accuracy(cam_id=[cam_id], cutoff=cutoff)
    print(f'|   | {orig_dict[cam_id]}')
    f,l,m = cam_flm(1, [cam_id])
    f,l,m = prune_flm(f,l,m, cutoff)
    # sampled_ind_full = kmeans_closest_classes_sample_ind(f,l,batch=cutoff)
    unique_classes = np.unique(l)
    num_classes = len(unique_classes)
    for batch in range(1,cutoff):
        sampled_ind_full = kmeans_closest_classes_sample_ind(f,l,batch=batch)
        # sampled_ind = sampled_ind_full[:batch*num_classes]
        nonsampled_ind = np.ones(l.shape[0]) == 1
        nonsampled_ind[sampled_ind] = False
        print(f'|   | {batch}')
        prediction_acc = 0
        for i in range(3):
            prediction_acc += get_prediction_accuracy(sampled_ind, nonsampled_ind,cam_id = [cam_id], cutoff=cutoff)#, batch=batch)
        prediction_acc /= 3
        print(f'|   | {prediction_acc}')
        cam_dict[cam_id].append(prediction_acc)


# In[93]:


# cam_ids = prune_cam_id()
# print(f'Total {len(cam_ids)} to check')
# cam_dict = {}
# orig_dict = {}
# cutoff = 25
# for cam_id in cam_ids:
#     print(f'| Cam ID {cam_id}')
#     cam_dict[cam_id] = []
#     orig_dict[cam_id] = get_original_accuracy(cam_id=[cam_id], cutoff=cutoff)
#     print(f'|   | {orig_dict[cam_id]}')
#     for batch in range(1,cutoff):
#         print(f'|   | {batch}')
#         prediction_acc = 0
#         for i in range(3):
#             prediction_acc += get_prediction_accuracy(cam_id = [cam_id], cutoff=cutoff, batch=batch)
#         prediction_acc /= 3
#         print(f'|   | {prediction_acc}')
#         cam_dict[cam_id].append(prediction_acc)


# In[55]:


import pickle

def get_dict(model):
    root_path = '/dccstor/hoo-misha-1/wilds/wilds/results/iwildcam'
    base_path = f'{root_path}/{model}'
    
    with open(f'{base_path}_cam_dict.pkl','rb') as file:
        cam_dict = pickle.load(file)

    with open(f'{base_path}_orig_dict.pkl','rb') as file:
        orig_dict = pickle.load(file)
    
    return cam_dict, orig_dict

def get_dict_path(root_path):
    cam_dict_path = f'{root_path}/kmeans_closest_batch_classes_cam_dict.pkl'
    orig_dict_path = f'{root_path}/kmeans_closest_batch_classes_orig_dict.pkl'
    
    with open(cam_dict_path,'rb') as file:
        cam_dict = pickle.load(file)

    with open(orig_dict_path,'rb') as file:
        orig_dict = pickle.load(file)
    
    return cam_dict, orig_dict


# In[56]:


cam_dict, orig_dict = get_dict_path('/dccstor/hoo-misha-1/wilds/WOODS/notebooks')


# In[57]:


cam_ids = prune_cam_id()


# In[96]:


from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt

def plot(cam_ind):
    predictions = cam_dict[cam_ids[cam_ind]]
    print(f'Original {orig_dict[cam_ids[cam_ind]]}')
    print(f'Max {max(predictions)}')
    metadata = np.load(f'{path_base}/resnet50_test_metadata.npy')
    unique_counts = np.unique(metadata[:,0],return_counts=True)
    ind = np.where(unique_counts[0] == cam_ids[cam_ind])
    print(f'With {unique_counts[1][ind]} data points pre-pruning')
    predictions = np.hstack((orig_dict[cam_ids[cam_ind]] , predictions))
    plt.plot(range(0,len(predictions)), predictions)
    
interact(plot, cam_ind=(0,len(cam_ids)));


# In[ ]:


from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt

good_inds = []
for i in range(len(cam_ids)):
    predictions = cam_dict[cam_ids[i]]
    if predictions[-1] > 0:
        good_inds.append(i)
        
def plot_2(cam_ind):
    cam_ind = good_inds[cam_ind]
    print(f'Camera id {cam_ids[cam_ind]}')
    predictions = cam_dict[cam_ids[cam_ind]]
    print(f'Original {orig_dict[cam_ids[cam_ind]]}')
    print(f'Max {max(predictions)}')
    metadata = np.load(f'{path_base}/resnet50_test_metadata.npy')
    unique_counts = np.unique(metadata[:,0],return_counts=True)
    ind = np.where(unique_counts[0] == cam_ids[cam_ind])
    print(f'With {unique_counts[1][ind]} data points pre-pruning')
    predictions = np.hstack((orig_dict[cam_ids[cam_ind]] , predictions))
    plt.plot(range(0,len(predictions)), predictions)


interact(plot_2, cam_ind=(0,len(good_inds)));


# In[60]:


def print_green(text, green=True, end='\n'):
    print(f'\x1b[{32 if green else 31}m{text}\x1b[0m', end=end)
def show_dist(cam_ind, cutoff=25):
    f,l,m = cam_flm(cam_id=[cam_ids[cam_ind]])
    unique_counts = np.unique(l, return_counts=True)
    print(f'Total of {sum(unique_counts[1] > cutoff)} classes over cutoff')
    print('[',end='')
    for y,c in zip(unique_counts[0], unique_counts[1]):
        print_green(f'{y}:{c}:{c/sum(unique_counts[1]):.2f}, ', c > cutoff, end='')
    print(']')

interact(show_dist, cam_ind=(0,len(cam_ids)-1), cutoff=(10,500));


# In[52]:


root_path = '/dccstor/hoo-misha-1/wilds/wilds/features/iwildcam'
models = list(os.listdir(root_path))

def plot_3(model_ind, cam_ind):
    global cam_dict, orig_dict
    model = models[model_ind]
    print(f'Using {model}')
    cam_dict, orig_dict = get_dict(model)
    plot_2(cam_ind)

interact(plot_3, model_ind=(0,len(models)-1), cam_ind=(0,len(good_inds)-1));



# In[ ]:


import pickle

with open('kmeans_closest_classes_cam_dict.pkl','wb') as file:
    pickle.dump(cam_dict, file)
    
with open('kmeans_closest_classes_orig_dict.pkl','wb') as file:
    pickle.dump(orig_dict, file)


# In[ ]:




