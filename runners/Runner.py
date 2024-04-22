import torch
import numpy as np
from models.TSCTM import TSCTM
import wandb


EPSILON = 0.01


def get_filled_topics_indices(theta: np.ndarray) -> np.ndarray:
    """
    Find the indices of the theta columns (i.e. the topics) that are
    filled with some documents.
    """

    return np.where(np.max(theta.T, axis=1) > EPSILON)[0]


def transition_frequency(theta: np.ndarray, theta_prev: np.ndarray) -> float:
    assert len(theta) == len(theta_prev)

    prev_labels = np.argmax(theta, axis=1)
    curr_labels = np.argmax(theta_prev, axis=1)

    return np.count_nonzero(prev_labels != curr_labels) / len(theta)


def v_measure(theta: np.ndarray, true_labels: np.ndarray)


class Runner():
    def __init__(self, config, device):
        self.config = config
        self.model = TSCTM(config)
        self.model = self.model.to(device)

        wandb.watch(self.model, log_freq=100)

    def train(self, train_loader):
        data_size = len(train_loader.dataset)
        optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate)

        theta_prev = None

        for epoch in range(1, self.config.num_epoch + 1):
            self.model.train()

            theta = [None] * data_size

            train_loss = 0
            contrastive_loss = 0
            for idx, train_data in enumerate(train_loader):
                bows = train_data['bow']

                batch_theta, batch_loss, batch_contrastive_loss = self.model(train_data)
                batch_theta = list(batch_theta.detach().cpu().numpy())

                for idx, theta_row in zip(train_data['id'], batch_theta):
                    # Reorder this shit
                    theta[idx] = theta_row

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.cpu() * len(bows)
                contrastive_loss += batch_contrastive_loss.cpu() * len(bows)

            # We have the ordered theta...
            theta = np.asarray(theta)

            freq = None
            if theta_prev is not None:
                freq = transition_frequency(theta, theta_prev)

            theta_prev = theta

            filled_topics = len(get_filled_topics_indices(theta))

            wandb.log({
                "epoch": epoch,
                "loss": train_loss / data_size,
                "contrastive_loss": contrastive_loss / data_size,
                "transition_frequency": freq,
                "filled_topics": filled_topics,
            })

            if epoch % 5 == 0:
                print('Epoch: {:03d}/{:03d} Loss: {:.3f}'.format(epoch, self.config.num_epoch, train_loss / data_size), end=' ')
                print('Contra_loss: {:.3f}'.format(contrastive_loss / data_size))

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
