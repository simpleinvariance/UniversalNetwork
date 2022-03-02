from argparse import ArgumentParser
from torch.cuda import set_device

from helpers.basic_classes import DataSet, PoolType
from experiment import Experiment


if __name__ == '__main__':
    parser = ArgumentParser()

    # DataSetArgs
    parser.add_argument("--dataset", dest="dataset", default=DataSet.ModelNet40, type=DataSet.from_string,
                        choices=list(DataSet), required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=32, type=int, required=False)

    # KroneckerArgs
    parser.add_argument("--n_neighbors", dest="n_neighbors", type=int, default=0, required=False)  # Best is 27
    parser.add_argument("--dynamic_knn", dest="dynamic_knn", action='store_true', required=False)

    # ReLUArgs
    parser.add_argument("--add_relus", dest="add_relus", action='store_true', required=False)
    parser.add_argument("--eps", dest="eps", type=float, default=1e-10, required=False)
    parser.add_argument("--share", dest="share", action='store_true', required=False)
    parser.add_argument("--negative_slope", dest="negative_slope", type=float, default=0.0, required=False)

    # GeneralHeadArgs
    parser.add_argument("--k", dest="k", type=int, default=2, required=False)
    parser.add_argument("--add_linears", dest="add_linears", action='store_true', required=False)
    parser.add_argument("--in_channel", dest="in_channel", type=int, default=128, required=False)
    parser.add_argument("--u_shape", dest="u_shape", action='store_true', required=False)
    parser.add_argument("--z_align", dest="z_align", action='store_true', required=False)
    parser.add_argument("--pool_type", dest="pool_type", default=PoolType.NONE, type=PoolType.from_string,
                        choices=list(PoolType), required=False)

    parser.add_argument("--drop_out", dest="drop_out", type=float, default=0.1, required=False)

    # TrainerArgs
    parser.add_argument("--epochs", dest="epochs", default=100, type=int, required=False)
    parser.add_argument("--lr", dest="lr", type=float, default=0.03, required=False)
    parser.add_argument("--additive_noise", dest="additive_noise", default=0.01, type=float, required=False)
    parser.add_argument("--scale_noise", dest="scale_noise", default=1.00, type=float, required=False)
    parser.add_argument("--SO3_train", dest="SO3_train", action='store_true', required=False)

    # result reproduction
    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)
    parser.add_argument('--gpu', type=int, required=False)

    # GPU Usage
    args = parser.parse_args()
    if args.gpu is not None:
        set_device(args.gpu)

    Experiment(args=args).run()
