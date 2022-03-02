import torch
from typing import List, Tuple

from helpers.train import prepare_raw_data, apply_noise
from helpers.basic_classes import DataSet, TrainerArgs, Task
from helpers.general import print_results_list

ROUND_DIGITS = 4


class Trainer(object):
    def __init__(self, model, optimizer, task: Task, dataset: DataSet, trainer_args: TrainerArgs, device):
        """
        Our trainer

        :param model:
        :param optimizer:
        :param task: Task
        :param dataset: DataSet
        :param trainer_args: TrainerArgs
        :param device:
        """
        self.model = model
        self.optimizer = optimizer
        self.task = task
        self.dataset = dataset
        self.trainer_args = trainer_args
        self.device = device

        self.loss = task.get_loss()

    def fit(self, train_loader, test_loader, epochs: int, batch_size: int):
        """
        Our fit method

        :param train_loader
        :param test_loader
        :param epochs: int
        :param batch_size: int
        """
        for epoch in range(0, epochs + 1):
            self.train(loader=train_loader, batch_size=batch_size)
            results_list = get_test_results(model=self.model, task=self.task, dataset=self.dataset,
                                            trainer_args=self.trainer_args, train_loader=train_loader,
                                            test_loader=test_loader, device=self.device)

            print_results_list(results_list=results_list, epoch=epoch)

    def train(self, loader, batch_size: int):
        """
        Our train method

        :param loader:
        :param batch_size: int
        """
        self.model.train()
        self.optimizer.zero_grad()

        for idx, data in enumerate(loader):
            data = prepare_raw_data(data=data, dataset=self.dataset, loader=loader,
                                    SO3_rotate=self.trainer_args.SO3_train).to(device=self.device)
            data.pos = apply_noise(x=data.pos, additive_noise=self.trainer_args.additive_noise,
                                   scale_noise=self.trainer_args.scale_noise)

            model_output = self.model(x=data.pos)
            self.loss(model_output, data.y).backward()

            # update
            if idx % batch_size == batch_size - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()


def get_test_results(model, task, dataset: DataSet, trainer_args: TrainerArgs, train_loader, test_loader, device)\
        -> List[Tuple[str, float]]:
    clas_or_seg = task in [Task.Classification]
    SO3_train_loss, SO3_train_acc = test(model=model, task=task, dataset=dataset,
                                         loader=train_loader, SO3_rotate=True, device=device)
    SO3_test_loss, SO3_test_acc = test(model=model, task=task, dataset=dataset,
                                       loader=test_loader, SO3_rotate=True, device=device)

    # saving results
    results_list = []
    if not trainer_args.SO3_train:
        Z_train_loss, Z_train_acc = test(model=model, task=task, dataset=dataset,
                                         loader=train_loader, SO3_rotate=False, device=device)
        Z_test_loss, Z_test_acc = test(model=model, task=task, dataset=dataset,
                                       loader=test_loader, SO3_rotate=False, device=device)
        results_list += [('Z Train Loss', Z_train_loss),
                         ('Z Test Loss', Z_test_loss)]
        if clas_or_seg:
            results_list += [('Z Train Acc', Z_train_acc),
                             ('Z Test Acc', Z_test_acc)]

    results_list += [('SO3 Train Loss', SO3_train_loss),
                     ('SO3 Test Loss', SO3_test_loss)]
    if clas_or_seg:
        results_list += [('SO3 Train Acc', SO3_train_acc),
                         ('SO3 Test Acc', SO3_test_acc)]

    return results_list


@torch.no_grad()
def test(model, task: Task, dataset: DataSet, loader, SO3_rotate: bool, device) -> Tuple[float, float]:
    """
    Our test method

    :param model
    :param task: Task
    :param dataset
    :param loader
    :param SO3_rotate: bool
    :param device
    :return: avg_loss: float
    :return: acc: float
    """
    model.eval()
    loss = task.get_loss()

    total_loss = correct = 0
    num_of_elements = 0 if task is Task.Segmentation else len(loader.dataset)
    for data in loader:
        data = prepare_raw_data(data=data, dataset=dataset, loader=loader,
                                SO3_rotate=SO3_rotate).to(device=device)
        model_output = model(x=data.pos)
        total_loss += loss(model_output, data.y).item()

        if task in [Task.Classification, Task.Segmentation]:
            correct += model_output.argmax(dim=1).eq(data.y).sum().item()

        if task is Task.Segmentation:
            num_of_elements += data.num_nodes

    return round(total_loss / len(loader), ROUND_DIGITS), round(correct / num_of_elements, ROUND_DIGITS)
