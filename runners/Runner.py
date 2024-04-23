import torch
import numpy as np
from models.TSCTM import TSCTM
import wandb
from sklearn.metrics import homogeneity_completeness_v_measure
from typing import Tuple, List
from utils.metrics_logger import MetricsLogger
from utils.TU import TU


EPSILON = 0.01


def get_filled_topics_indices(theta: np.ndarray) -> np.ndarray:
    """
    Find the indices of the theta columns (i.e. the topics) that are
    filled with some documents.
    """

    return np.where(np.max(theta.T, axis=1) > EPSILON)[0]


def cut_theta(theta: np.ndarray) -> np.ndarray:
    """
    Cut theta, keeping the columns that are filled with some documents.
    """

    return theta.T[get_filled_topics_indices(theta)].T


def cut_beta(beta: np.ndarray, filled_topics_indices: np.ndarray) -> np.ndarray:
    """
    Cut beta, keeping the columns that are filled with some documents.
    """

    return beta[filled_topics_indices]


def c_transition_frequency(theta: np.ndarray, theta_prev: np.ndarray) -> float:
    assert len(theta) == len(theta_prev)

    prev_labels = np.argmax(theta, axis=1)
    curr_labels = np.argmax(theta_prev, axis=1)

    return np.count_nonzero(prev_labels != curr_labels) / len(theta)


def supervised_scores(theta: np.ndarray, test_labels: np.ndarray) -> Tuple[float, float, float]:
    pred_labels = np.argmax(theta, axis=1)

    return homogeneity_completeness_v_measure(test_labels, pred_labels)


def get_top_words(beta: np.ndarray, vocab: List[str], topk=15) -> List[str]:
    topic_str_list = []

    for topic_dist in beta:
        topic_words = np.asarray(vocab)[np.argsort(topic_dist)][:-(topk + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)

    return topic_str_list


class Runner():
    def __init__(self, config, device, vocab: List[str], ml: MetricsLogger):
        self.config = config
        self.model = TSCTM(config, ml)
        self.model = self.model.to(device)
        self.vocab = vocab
        self.ml = ml

        wandb.watch(self.model, log_freq=100)

    def train(self, train_loader):
        data_size = len(train_loader.dataset)
        optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate)

        theta_prev = None

        for epoch in range(1, self.config.num_epoch + 1):
            self.model.train()

            theta = [None] * data_size
            test_labels = [None] * data_size

            train_loss = 0
            contrastive_loss = 0
            for idx, train_data in enumerate(train_loader):
                bows = train_data['bow']

                batch_theta, batch_loss, batch_contrastive_loss = self.model(train_data)
                batch_theta = list(batch_theta.detach().cpu().numpy())

                for idx, theta_row in zip(train_data['id'], batch_theta):
                    # Reorder this shit
                    theta[idx] = theta_row

                if 'label' in train_data:
                    for idx, test_label in zip(train_data['id'], train_data['label']):
                        # Reorder this shit
                        test_labels[idx] = test_label

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.cpu() * len(bows)
                contrastive_loss += batch_contrastive_loss.cpu() * len(bows)

            # We have the ordered theta...
            theta = np.asarray(theta)
            beta = self.model.get_beta().detach().cpu().numpy()

            if test_labels[0] is not None:
                test_labels = np.asarray(test_labels)

            transition_frequency = None
            if theta_prev is not None:
                transition_frequency = c_transition_frequency(theta, theta_prev)

            filled_topics_indices = get_filled_topics_indices(theta)
            theta_cut = cut_theta(theta)
            beta_cut = cut_beta(beta, filled_topics_indices)

            topk_words = get_top_words(beta_cut, self.vocab)
            tu = TU(topk_words)

            filled_topics = len(filled_topics_indices)

            d = {
                "transition_frequency": transition_frequency,
                "filled_topics": filled_topics,
                "tu": tu,
            }

            if test_labels[0] is not None:
                h, c, v = supervised_scores(theta_cut, test_labels)

                d['h'] = h
                d['c'] = c
                d['v'] = v

            theta_prev = theta

            self.ml.log_dict(d)

            print('Epoch: {:03d}/{:03d} Loss: {:.3f}'.format(epoch, self.config.num_epoch, train_loss / data_size), end=' ')
            print('Contra_loss: {:.3f}'.format(contrastive_loss / data_size))

            self.ml.end_epoch()

        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def test(self, inputs):
        data_size = inputs.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.config.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_inputs = inputs[idx]
                batch_theta = self.model.get_theta(batch_inputs)
                theta += list(batch_theta.detach().cpu().numpy())

        theta = np.asarray(theta)
        return theta
