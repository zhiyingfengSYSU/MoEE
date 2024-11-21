"""Core functions of federate learning."""
import argparse
import copy

import numpy as np
from advertorch.attacks import LinfPGDAttack
from torch import nn

from federated.aggregation import ModelAccumulator, SlimmableModelAccumulator
from nets.slimmable_models import get_slim_ratios_from_str, parse_lognorm_slim_schedule
from utils.utils import shuffle_sampler, str2bool


class _Federation_HeterDataset:
    """A helper class for federated data creation.
    Use `add_argument` to setup ArgumentParser and then use parsed args to init the class.
    """
    _model_accum: ModelAccumulator

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        # data
        parser.add_argument('--percent', type=float, default=1,
                            help='percentage of dataset for training')
        parser.add_argument('--val_ratio', type=float, default=0.1,
                            help='ratio of train set for validation')
        parser.add_argument('--batch', type=int, default=32, help='batch size')
        parser.add_argument('--test_batch', type=int, default=128, help='batch size for test')

        # federated
        parser.add_argument('--pd_nuser', type=int, default=10, help='#users per domain.')
        parser.add_argument('--pr_nuser', type=int, default=-1, help='#users per comm round [default: all]')
        parser.add_argument('--pu_nclass', type=int, default=-1, help='#class per user. -1 or 0: all')
        parser.add_argument('--domain_order', choices=list(range(5)), type=int, default=0,
                            help='select the order of domains')
        parser.add_argument('--partition_mode', choices=['uni', 'dir'], type=str.lower, default='uni',
                            help='the mode when splitting domain data into users: uni - uniform '
                                 'distribution (all user have the same #samples); dir - Dirichlet'
                                 ' distribution (non-iid sample sizes)')
        parser.add_argument('--con_test_cls', action='store_true',
                            help='Ensure the test classes are the same training for a client. '
                                 'Meanwhile, make test sets are uniformly splitted for clients. '
                                 'Mainly influence class-niid settings.')
        parser.add_argument('--local_fc', action='store_true', help='use local FC layer.')

    @classmethod
    def render_run_name(cls, args):
        run_name = f'__pd_nuser_{args.pd_nuser}'
        if args.percent != 0.3: run_name += f'__pct_{args.percent}'
        if args.pu_nclass > 0: run_name += f"__pu_nclass_{args.pu_nclass}"
        if args.pr_nuser != -1: run_name += f'__pr_nuser_{args.pr_nuser}'
        if args.domain_order != 0: run_name += f'__do_{args.domain_order}'
        if args.partition_mode != 'uni': run_name += f'__part_md_{args.partition_mode}'
        if args.con_test_cls: run_name += '__ctc'
        if args.local_fc: run_name += '__lfc'
        return run_name

    def __init__(self, data, num_users, args, hard, distributionRate_labels_by_users):
        self.args = args

        # Prepare Data
        num_classes = 10
        if data == 'Digits':
            from utils.data_utils import DigitsDataset
            from utils.data_loader import prepare_digits_data
            prepare_data = prepare_digits_data
            DataClass = DigitsDataset
        elif data == 'DomainNet':
            from utils.data_utils import DomainNetDataset
            from utils.data_loader import prepare_domainnet_data
            prepare_data = prepare_domainnet_data
            DataClass = DomainNetDataset
        elif data == 'cifar10':
            from utils.data_utils import CifarDataset
            from utils.data_loader import prepare_cifar_data
            prepare_data = prepare_cifar_data
            DataClass = CifarDataset
        elif data == 'stl':
            from utils.data_utils import STLDataset
            from utils.data_loader import prepare_stl_data
            prepare_data = prepare_stl_data
            DataClass = STLDataset
        elif data == 'cifar100':
            from utils.data_utils import CifarDataset100
            from utils.data_loader import prepare_cifar100_data
            prepare_data = prepare_cifar100_data
            DataClass = CifarDataset100
            num_classes = 100
        elif data == 'tin':
            from utils.data_utils import TinyImageNet
            from utils.data_loader import prepare_imagenet_data
            prepare_data = prepare_imagenet_data
            DataClass = TinyImageNet
            num_classes = 200
        elif data == 'ImageNet':
            from utils.data_utils import ImageNet12
            from utils.data_loader import prepare_ImageNet_data
            prepare_data = prepare_ImageNet_data
            DataClass = ImageNet12
            num_classes = 12
        elif data == 'mnist':
            from utils.data_utils import MNISTDataset
            from utils.data_loader import prepare_mnist_data
            prepare_data = prepare_mnist_data
            DataClass = MNISTDataset
            num_classes = 10
        elif data == 'fashion_mnist':
            from utils.data_utils import FASHION_MNISTDataset
            from utils.data_loader import prepare_fashionmnist_data
            prepare_data = prepare_fashionmnist_data
            DataClass = FASHION_MNISTDataset
            num_classes = 10
        elif data == 'SVHN':
            from utils.data_utils import SVHNDataset
            from utils.data_loader import prepare_SVHN_data
            prepare_data = prepare_SVHN_data
            DataClass = SVHNDataset
            num_classes = 10
        else:
            raise ValueError(f"Unknown dataset: {data}")
        all_domains = DataClass.resorted_domains[args.domain_order]

        train_loaders, val_loaders, test_loaders, clients, distributionRate_labels_users = prepare_data(
            args, domains=all_domains,
            n_user_per_domain=num_users,
            n_class_per_user=args.pu_nclass,
            partition_seed=args.seed + 1,
            partition_mode=args.partition_mode,
            val_ratio=args.val_ratio,
            eq_domain_train_size=args.partition_mode == 'uni',
            consistent_test_class=args.con_test_cls,
            hard=hard,
            distributionRate_labels_by_users=distributionRate_labels_by_users
        )
        clients = [c + ' ' + ('noised' if hasattr(args, 'adv_lmbd') and args.adv_lmbd > 0.
                              else 'clean') for c in clients]

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.distributionRate_labels_users = distributionRate_labels_users
        self.clients = clients
        self.num_classes = num_classes
        self.all_domains = all_domains

        # Setup fed
        self.client_num = len(self.clients)
        client_weights = [len(tl.dataset) for tl in train_loaders]
        self.client_weights = [w / sum(client_weights) for w in client_weights]

        pr_nuser = args.pr_nuser if args.pr_nuser > 0 else self.client_num
        self.args.pr_nuser = pr_nuser
        self.client_sampler = UserSampler([i for i in range(self.client_num)], pr_nuser, mode='uni')

    def get_data(self):
        return self.train_loaders, self.val_loaders, self.test_loaders, self.distributionRate_labels_users

class _Federation_HeterDataset_MoE:
    """A helper class for federated data creation.
    Use `add_argument` to setup ArgumentParser and then use parsed args to init the class.
    """
    _model_accum: ModelAccumulator

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        # data
        parser.add_argument('--percent', type=float, default=1,
                            help='percentage of dataset for training')
        parser.add_argument('--val_ratio', type=float, default=0.1,
                            help='ratio of train set for validation')
        parser.add_argument('--batch', type=int, default=32, help='batch size')
        parser.add_argument('--test_batch', type=int, default=128, help='batch size for test')

        # federated
        parser.add_argument('--pd_nuser', type=int, default=10, help='#users per domain.')
        parser.add_argument('--pr_nuser', type=int, default=-1, help='#users per comm round '
                                                                     '[default: all]')
        parser.add_argument('--pu_nclass', type=int, default=-1, help='#class per user. -1 or 0: all')
        parser.add_argument('--domain_order', choices=list(range(5)), type=int, default=0,
                            help='select the order of domains')
        parser.add_argument('--partition_mode', choices=['uni', 'dir'], type=str.lower, default='uni',
                            help='the mode when splitting domain data into users: uni - uniform '
                                 'distribution (all user have the same #samples); dir - Dirichlet'
                                 ' distribution (non-iid sample sizes)')
        parser.add_argument('--con_test_cls', action='store_true',
                            help='Ensure the test classes are the same training for a client. '
                                 'Meanwhile, make test sets are uniformly splitted for clients. '
                                 'Mainly influence class-niid settings.')
        parser.add_argument('--local_fc', action='store_true', help='use local FC layer.')

    @classmethod
    def render_run_name(cls, args):
        run_name = f'__pd_nuser_{args.pd_nuser}'
        if args.percent != 0.3: run_name += f'__pct_{args.percent}'
        if args.pu_nclass > 0: run_name += f"__pu_nclass_{args.pu_nclass}"
        if args.pr_nuser != -1: run_name += f'__pr_nuser_{args.pr_nuser}'
        if args.domain_order != 0: run_name += f'__do_{args.domain_order}'
        if args.partition_mode != 'uni': run_name += f'__part_md_{args.partition_mode}'
        if args.con_test_cls: run_name += '__ctc'
        if args.local_fc: run_name += '__lfc'
        return run_name

    def __init__(self, data, num_users, args, hard, distributionRate_labels_by_users):
        self.args = args

        # Prepare Data
        num_classes = 10
        if data == 'Digits':
            from utils.data_utils import DigitsDataset
            from utils.data_loader import prepare_digits_data
            prepare_data = prepare_digits_data
            DataClass = DigitsDataset
        elif data == 'DomainNet':
            from utils.data_utils import DomainNetDataset
            from utils.data_loader import prepare_domainnet_data
            prepare_data = prepare_domainnet_data
            DataClass = DomainNetDataset
        elif data == 'cifar10':
            from utils.data_utils import CifarDataset
            from utils.data_loader import prepare_cifar_data
            prepare_data = prepare_cifar_data
            DataClass = CifarDataset
        elif data == 'stl':
            from utils.data_utils import STLDataset
            from utils.data_loader import prepare_stl_data
            prepare_data = prepare_stl_data
            DataClass = STLDataset
        elif data == 'cifar100':
            from utils.data_utils import CifarDataset100
            from utils.data_loader import prepare_cifar100_data
            prepare_data = prepare_cifar100_data
            DataClass = CifarDataset100
            num_classes = 100
        elif data == 'tin':
            from utils.data_utils import TinyImageNet
            from utils.data_loader import prepare_imagenet_data
            prepare_data = prepare_imagenet_data
            DataClass = TinyImageNet
            num_classes = 200
        elif data == 'ImageNet':
            from utils.data_utils import ImageNet12
            from utils.data_loader import prepare_ImageNet_data
            prepare_data = prepare_ImageNet_data
            DataClass = ImageNet12
            num_classes = 12
        elif data == 'mnist':
            from utils.data_utils import MNISTDataset
            from utils.data_loader import prepare_mnist_data
            prepare_data = prepare_mnist_data
            DataClass = MNISTDataset
        elif data == 'fashion_mnist':
            from utils.data_utils import FASHION_MNISTDataset
            from utils.data_loader import prepare_fashionmnist_data
            prepare_data = prepare_fashionmnist_data
            DataClass = FASHION_MNISTDataset
            num_classes = 10
        elif data == 'SVHN':
            from utils.data_utils import SVHNDataset
            from utils.data_loader import prepare_SVHN_data
            prepare_data = prepare_SVHN_data
            DataClass = SVHNDataset
            num_classes = 10
        else:
            raise ValueError(f"Unknown dataset: {data}")
        all_domains = DataClass.resorted_domains[args.domain_order]

        train_loaders, val_loaders, test_loaders, clients, distributionRate_labels_users = prepare_data(
            args, domains=all_domains,
            n_user_per_domain=num_users,
            n_class_per_user=args.pu_nclass+2,
            partition_seed=args.seed + 1,
            partition_mode=args.partition_mode,
            val_ratio=args.val_ratio,
            eq_domain_train_size=args.partition_mode == 'uni',
            consistent_test_class=args.con_test_cls,
            hard=hard,
            distributionRate_labels_by_users=distributionRate_labels_by_users
        )
        clients = [c + ' ' + ('noised' if hasattr(args, 'adv_lmbd') and args.adv_lmbd > 0.
                              else 'clean') for c in clients]

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.distributionRate_labels_users = distributionRate_labels_users
        self.clients = clients
        self.num_classes = num_classes
        self.all_domains = all_domains

        # Setup fed
        self.client_num = len(self.clients)
        client_weights = [len(tl.dataset) for tl in train_loaders]
        self.client_weights = [w / sum(client_weights) for w in client_weights]

        pr_nuser = args.pr_nuser if args.pr_nuser > 0 else self.client_num
        self.args.pr_nuser = pr_nuser
        self.client_sampler = UserSampler([i for i in range(self.client_num)], pr_nuser, mode='uni')

    def get_data(self):
        return self.train_loaders, self.val_loaders, self.test_loaders, self.distributionRate_labels_users

class UserSampler(object):
    def __init__(self, users, select_nuser, mode='all'):
        self.users = users
        self.total_num_user = len(users)
        self.select_nuser = select_nuser
        self.mode = mode
        if mode == 'all':
            assert select_nuser == self.total_num_user, "Conflict config: Select too few users."

    def iter(self):
        if self.mode == 'all' or self.select_nuser == self.total_num_user:
            sel = np.arange(len(self.users))
        elif self.mode == 'uni':
            sel = np.random.choice(self.total_num_user, self.select_nuser, replace=False)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        for i in sel:
            yield self.users[i]


class AdversaryCreator(object):
    """A factory producing adversary.

    Args:
        attack: Name. MIA for MomentumIterativeAttack with Linf norm. LSA for LocalSearchAttack.
        eps: Constraint on the distortion norm
        steps: Number of attack steps
    """
    supported_adv = ['LinfPGD', 'LinfPGD20', 'LinfPGD20_eps16', 'LinfPGD100','LinfPGD100_eps16',
                     'LinfPGD4_eps4', 'LinfPGD3_eps4', 'LinfPGD7_eps4',
                     ]

    def __init__(self, attack: str, **kwargs):
        self.attack = attack
        if '_eps' in self.attack:
            self.attack, default_eps = self.attack.split('_eps')
            self.eps = kwargs.setdefault('eps', int(default_eps))
        else:
            self.eps = kwargs.setdefault('eps', 8.)
        if self.attack.startswith('LinfPGD') and self.attack[len('LinfPGD'):].isdigit():
            assert 'steps' not in kwargs, "The steps is set by the attack name while " \
                                          "found additional set in kwargs."
            self.steps = int(self.attack[len('LinfPGD'):])
        elif self.attack.startswith('MIA') and self.attack[len('MIA'):].isdigit():
            assert 'steps' not in kwargs, "The steps is set by the attack name while " \
                                          "found additional set in kwargs."
            self.steps = int(self.attack[len('MIA'):])
        else:
            self.steps = kwargs.setdefault('steps', 7)

    def __call__(self, model):
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        if self.attack.startswith('LinfPGD'):
            adv = LinfPGDAttack(
                model, loss_fn=loss_fn, eps=self.eps / 255,
                nb_iter=self.steps,
                eps_iter=min(self.eps / 255 * 1.25, self.eps / 255 + 4. / 255) / self.steps,
                rand_init=True,
                clip_min=0.0, clip_max=1.0,
                targeted=False)
        elif self.attack == 'none':
            adv = None
        else:
            raise ValueError(f"attack: {self.attack}")
        return adv
