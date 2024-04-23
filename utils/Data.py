import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import Dataset
from utils import data_utils


class TextData(Dataset):
    def __init__(self, data_dir, device, aug_option_list=None, use_aug=False):
        dataset_file = 'train_texts.txt'
        labels_file = 'train_labels.txt'

        self.train_texts = data_utils.read_text(os.path.join(data_dir, '{}'.format(dataset_file)))
        train_size = len(self.train_texts)

        self.use_aug = use_aug
        self.aug_option_list = aug_option_list

        vectorizer = CountVectorizer(min_df=5)
        self.train_bow = vectorizer.fit_transform(self.train_texts).toarray().astype('float32')

        self.train_vocab = vectorizer.get_feature_names_out()

        if aug_option_list:
            print('===>Info: reading augmentation data...')

            self.num_contrast = len(aug_option_list)
            self.aug_texts_list = list()
            for aug_option in aug_option_list:
                aug_text_path = os.path.join(data_dir, '{}_{}'.format(dataset_file, aug_option))
                print('===>reading {}'.format(aug_text_path))
                self.aug_texts_list.append(data_utils.read_text(aug_text_path))

            self.combined_train_texts = np.concatenate((np.asarray(self.train_texts), np.asarray(self.aug_texts_list).flatten()))
            combined_train_bow = vectorizer.fit_transform(self.combined_train_texts).toarray().astype('float32')

            if self.use_aug:
                self.train_bow = combined_train_bow[:train_size]
                self.contrast_bow_list = np.array_split(combined_train_bow[train_size:], self.num_contrast)
                self.contrast_bow_list = [torch.tensor(bow).to(device) for bow in self.contrast_bow_list]

            else:
                self.train_bow = combined_train_bow

        self.train_bow = torch.tensor(self.train_bow).to(device)
        self.vocab = vectorizer.get_feature_names_out()

        labels_path = os.path.join(data_dir, labels_file)
        if os.path.isfile(labels_path):
            self.labels = np.asarray([int(l) for l in open(labels_path, 'r').read().splitlines()])

    def __len__(self):
        return len(self.train_bow)

    def __getitem__(self, idx):
        if self.use_aug:
            # TODO: update this code
            exit(69)
            return [self.train_bow[idx]] + [bow[idx] for bow in self.contrast_bow_list]
        else:
            d = {
                "id": idx,
                "bow": self.train_bow[idx],
            }

            if hasattr(self, 'labels'):
                d['label'] = self.labels[idx]

            return d
