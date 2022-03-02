import torch
import os.path as osp

from helpers.git_path import get_git_path
from helpers.general import record_args, set_seed, print_results_list
from trainer import Trainer, get_test_results
from helpers.basic_classes import KroneckerArgs, ReLUArgs, GeneralHeadArgs, TrainerArgs
from model import Model


class Experiment(object):
    """
    The run method is the main usage of this class

    :param args
    """
    def __init__(self, args):
        assert args.scale_noise >= 1.0, "scale_ratio should be larger or equal to 1.0"
        assert args.n_neighbors >= 0, "n_neighbors should be an integer that is larger or equal to 0"

        self.args = args

        # DataSetArgs
        self.dataset = args.dataset
        self.batch_size = args.batch_size

        # KroneckerArgs
        self.n_neighbors = args.n_neighbors
        self.dynamic_knn = args.dynamic_knn

        # ReLUArgs
        self.add_relus = args.add_relus
        self.eps = args.eps
        self.share = args.share
        self.negative_slope = args.negative_slope

        # GeneralHeadArgs
        self.k = args.k
        self.in_channel = args.in_channel
        self.add_linears = args.add_linears
        self.u_shape = args.u_shape
        self.z_align = args.z_align
        self.pool_type = args.pool_type

        self.drop_out = args.drop_out

        # TrainerArgs
        self.epochs = args.epochs
        self.lr = args.lr
        self.additive_noise = args.additive_noise
        self.scale_noise = args.scale_noise
        self.SO3_train = args.SO3_train

        # result reproduction
        self.seed = args.seed
        set_seed(seed=args.seed)

        print(f'######################## STARTING EXPERIMENT ########################')
        self.name_of_run = record_args(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        """
        The method is the main usage of this class
        """
        # load data
        train_loader, test_loader, out_channel = self.dataset.load(z_align=self.z_align)
        task = self.dataset.get_task()

        # prepare the different arguments
        kronecker_args = KroneckerArgs(n_neighbors=self.n_neighbors, dynamic_knn=self.dynamic_knn)
        relu_args = ReLUArgs(add=self.add_relus, eps=self.eps, share=self.share, negative_slope=self.negative_slope)
        general_head_args = GeneralHeadArgs(k=self.k, in_channel=self.in_channel, add_linears=self.add_linears,
                                            u_shape=self.u_shape, z_align=self.z_align, pool_type=self.pool_type)
        trainer_args = TrainerArgs(epochs=self.epochs, lr=self.lr, additive_noise=self.additive_noise,
                                   scale_noise=self.scale_noise, SO3_train=self.SO3_train)

        # create model
        model = Model(task=task, kronecker_args=kronecker_args, relu_args=relu_args,
                      general_head_args=general_head_args, drop_out=self.drop_out, out_channel=out_channel)
        model = model.to(device=self.device)

        # load pre-trained model
        model_path = osp.join(get_git_path(), self.name_of_run + '.pt')
        if osp.exists(model_path):
            # load model
            print('######################## Loading Model ########################', flush=True)
            model.load_state_dict(torch.load(model_path))
        else:
            # train model
            print('######################## Training Model ########################', flush=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=trainer_args.lr / self.batch_size)
            trainer = Trainer(task=task, model=model, optimizer=optimizer, dataset=self.dataset,
                              trainer_args=trainer_args, device=self.device)
            trainer.fit(train_loader=train_loader, test_loader=test_loader, epochs=self.epochs,
                        batch_size=self.batch_size)

        # Model Summary
        print('\n######################## Final Results ########################', flush=True)
        results_list = get_test_results(model=model, task=task, dataset=self.dataset, trainer_args=trainer_args,
                                        train_loader=train_loader, test_loader=test_loader, device=self.device)
        print_results_list(results_list=results_list)

        # save model
        if not osp.exists(model_path):
            torch.save(trainer.model.state_dict(), model_path)
