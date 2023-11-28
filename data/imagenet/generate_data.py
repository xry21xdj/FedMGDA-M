import os
import argparse
import pickle
import urllib.request
import zipfile

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

# Assuming you have the utility functions for splitting in the same location
from utils import split_dataset_by_labels, pathological_non_iid_split

#%%

ALPHA = .4
N_CLASSES = 200
N_COMPONENTS = 3
SEED = 1234
RAW_DATA_PATH = "raw_data/"
PATH = "all_data/"
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_and_unzip_dataset():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    zip_path = os.path.join(RAW_DATA_PATH, "tiny-imagenet-200.zip")

    # Download Tiny ImageNet
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)

    # Unzip the dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_PATH)


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True)
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as in'
             '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
             'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters;',
        type=int,
        default=N_COMPONENTS
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;',
        type=float,
        default=ALPHA
    )
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--val_frac',
        help='fraction of validation set (from train set); default: 0.0;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=SEED
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    download_and_unzip_dataset()

    transform = Compose([
        ToTensor(),
        Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))  # Tiny ImageNet normalization constants
    ])
    
    train_dataset = ImageFolder(root=os.path.join(RAW_DATA_PATH, 'tiny-imagenet-200/train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(RAW_DATA_PATH, 'tiny-imagenet-200/val'), transform=transform)
    dataset = ConcatDataset([train_dataset, val_dataset])

    # 获取标签类别信息
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 打印类别标签信息
    print("Class to Index Mapping:")
    print(class_to_idx)
    
    print("\nIndex to Class Mapping:")
    print(idx_to_class)
    
    # Define the path to your ImageNet (or Tiny ImageNet) dataset
    # dataset_path =  os.path.join("raw_data")
    # assert os.path.isdir(dataset_path), "Download imagenet dataset!!"
    
    # # Load the dataset
    # imagenet_dataset = ImageFolder(dataset_path)
    
    # # In the ImageFolder dataset, the data is a tuple of (image, class_index).
    # # Therefore, we split the data and targets.
    # data = [sample[0] for sample in imagenet_dataset]
    # targets = [sample[1] for sample in imagenet_dataset]
    # print('target=',targets)
    
    
    if args.pathological_split:
        clients_indices =\
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )
    else:
        clients_indices = \
            split_dataset_by_labels(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed
            )

    if args.test_tasks_frac > 0:
        train_clients_indices, test_clients_indices = \
            train_test_split(clients_indices, test_size=args.test_tasks_frac, random_state=args.seed)
    else:
        train_clients_indices, test_clients_indices = clients_indices, []

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            if len(indices) == 0:
                continue

            client_path = os.path.join(PATH, mode, "task_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)

            train_indices, test_indices =\
                train_test_split(
                    indices,
                    train_size=args.tr_frac,
                    random_state=args.seed
                )

            if args.val_frac > 0:
                train_indices, val_indices = \
                    train_test_split(
                        train_indices,
                        train_size=1.-args.val_frac,
                        random_state=args.seed
                    )

                save_data(val_indices, os.path.join(client_path, "val.pkl"))

            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(test_indices, os.path.join(client_path, "test.pkl"))


if __name__ == "__main__":
    main()
