# coding: UTF-8
import os
import time
import json
import pickle

import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from feature import FeatureExtractor


class VRD(Dataset):
    def __init__(self, dataset_root, split, cache_root='cache'):

        self.dataset_root = dataset_root
        self.is_testing = split == 'test'
        self.pre_categories = json.load(open(os.path.join(dataset_root, 'predicates.json')))
        self.obj_categories = json.load(open(os.path.join(dataset_root, 'objects.json')))

        if not os.path.exists(cache_root):
            os.makedirs(cache_root)

        if split == 'test':
            self.features, self.gt_labels = self.__prepare_testing_data__(cache_root)
        elif split == 'train' or split == 'val':
            self.features, self.gt_labels = self.__prepare_training_data__(split, cache_root)
        else:
            print('split in [train, val, test]')
            raise ValueError

        self.feature_len = self.features.shape[1]

    def pre_category_num(self):
        return len(self.pre_categories)

    def obj_category_num(self):
        return len(self.obj_categories)

    def __prepare_training_data__(self, split, cache_root):
        """
        准备特征文件和标签文件
        特征文件包含一个numpy浮点型二维矩阵，N x L，N为样本总数，L为特征长度
        标签文件包含一个numpy二值型二维矩阵，N x C，N为样本总数，C为关系类别数
        :param split: train or val
        :param cache_root: save root
        :return: features, labels
        """
        feature_path = os.path.join(cache_root, '%s_features.bin' % split)
        label_path = os.path.join(cache_root, '%s_labels.bin' % split)
        if not os.path.exists(feature_path) or not os.path.exists(label_path):
            print('Extracting features for %s set ...' % split)
            time.sleep(2)

            imgs_path = os.path.join(self.dataset_root, '%s_images' % split)
            ano_path = os.path.join(self.dataset_root, 'annotations_%s.json' % split)
            features_builder = []
            gt_labels_builder = []
            feature_extractor = FeatureExtractor()
            with open(ano_path, 'r') as f:
                annotation_all = json.load(f)
            file_list = dict()
            for root, dir, files in os.walk(imgs_path):
                for file in files:
                    file_list[file] = os.path.join(root, file)
            for file in tqdm(file_list):
                ano = annotation_all[file]
                samples_info = []
                labels = []
                for sample in ano:
                    gt_predicates = sample['predicate']
                    gt_object_id = sample['object']['category']
                    gt_object_loc = sample['object']['bbox']
                    gt_subject_id = sample['subject']['category']
                    gt_subject_loc = sample['subject']['bbox']
                    samples_info.append(gt_subject_loc + [gt_subject_id] + gt_object_loc + [gt_object_id])
                    predicates = np.zeros(self.pre_category_num())
                    for p in gt_predicates:
                        predicates[p] = 1
                    labels.append(predicates.tolist())
                feature = feature_extractor.extract_features(cv2.imread(file_list[file]), samples_info)
                features_builder = features_builder + feature.tolist()
                gt_labels_builder = gt_labels_builder + labels
            features = np.array(features_builder)
            gt_labels = np.array(gt_labels_builder)
            with open(feature_path, 'wb') as fw:
                pickle.dump(features, fw)
            with open(label_path, 'wb') as fw:
                pickle.dump(gt_labels, fw)
        else:
            print('Loading data ...')
            with open(feature_path, 'rb') as f:
                features = pickle.load(f)
            with open(label_path, 'rb') as f:
                gt_labels = pickle.load(f)

        return features, gt_labels

    def __prepare_testing_data__(self, cache_root):
        """
        准备特征文件
        特征文件包含一个numpy浮点型二维矩阵，N x L，N为样本总数，L为特征长度
        :param cache_root: save root
        :return: features, labels=None
        """
        feature_path = os.path.join(cache_root, 'test_features.bin')
        if not os.path.exists(feature_path):
            print('Extracting features for test set ...')
            time.sleep(2)

            imgs_path = os.path.join(self.dataset_root, 'test_images')
            ano_path = os.path.join(self.dataset_root, 'annotations_test_so.json')
            features_builder = []
            feature_extractor = FeatureExtractor()
            with open(ano_path, 'r') as f:
                annotation_all = json.load(f)
            file_list = dict()
            for root, dir, files in os.walk(imgs_path):
                for file in files:
                    file_list[file] = os.path.join(root, file)
            for file in tqdm(file_list):
                ano = annotation_all[file]
                samples_info = []
                for sample in ano:
                    gt_object_id = sample['object']['category']
                    gt_object_loc = sample['object']['bbox']
                    gt_subject_id = sample['subject']['category']
                    gt_subject_loc = sample['subject']['bbox']
                    samples_info.append(gt_subject_loc + [gt_subject_id] + gt_object_loc + [gt_object_id])
                feature = feature_extractor.extract_features(cv2.imread(file_list[file]), samples_info)
                features_builder = features_builder + feature.tolist()
            features = np.array(features_builder)
            with open(feature_path, 'wb') as fw:
                pickle.dump(features, fw)
        else:
            print('Loading data ...')
            with open(feature_path, 'rb') as f:
                features = pickle.load(f)

        return features, None

    def __getitem__(self, item):
        if self.is_testing:
            return self.features[item], 0
        else:
            return self.features[item], self.gt_labels[item]

    def __len__(self):
        return self.features.shape[0]

    def len(self):
        return self.features.shape[0]
