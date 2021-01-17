import os
import lmdb
import io
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils.map_imagenet import label2num

def get_train_dataloader(data_path, batchsize, num_workers, distributed=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                            ])
    train_dataset = LMDBDatabase(lmdb_path=data_path, mode='train', transform=transform)
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=batchsize, 
                            shuffle=(train_sampler is None), num_workers=num_workers, 
                            pin_memory=True, sampler=train_sampler)
    return train_loader

def get_val_dataloader(data_path, batchsize, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                            ])
    val_dataset = LMDBDatabase(lmdb_path=data_path, mode='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return val_loader

lmdb_path = ' ' # YOUR LMDB PATH
db_val_name = 'ILSVRC2012_img_val.lmdb'
db_train_name = 'ILSVRC2012_img_train.lmdb'

class LMDBDatabase(Dataset):
    """A class for LMDB database.
    Args:
    @param lmdb: str, the source LMDB database file.
    """
    def __init__(self, lmdb_path, mode='train', transform=None):
        super().__init__()
        if mode is 'train':
            lmdb_file = os.path.join(lmdb_path, db_train_name)
        elif mode is 'val':
            lmdb_file = os.path.join(lmdb_path, db_val_name) 
        assert isinstance(lmdb_file, str)
        assert os.path.isfile(lmdb_file)
        self.db_binary = lmdb_file
        self.db_env = lmdb.open(lmdb_file, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.db_env.begin(write=False) as txn:
            # load meta: __len__
            db_length = txn.get(b'__len__')
            if db_length is None:
                raise ValueError('LMDB database must have __len__ item which indicates number of records in the database.')
            self.db_length = int().from_bytes(db_length, 'big') # ATTENTION! Do save __len__ as bytes in big version when making the database
            # load meta: __key__
            db_keys = txn.get(b'__keys__')
            if db_keys is None:
                raise ValueError('LMDB database must have __keys__ item which holds the keys for retrieving record.')
            self.db_keys = pickle.loads(db_keys)
            # load meta: __files__
            db_files = txn.get(b'__files__')
            if db_files is None:
                raise ValueError('LMDB database must have __files__ item which holds the paths of original data files.')
            self.db_files = pickle.loads(db_files)
        assert self.db_length == len(self.db_keys)
        self.db_iter_idx = -1
        self.transform = transform
    
    def __len__(self):
        return self.db_length
    
    def __repr__(self):
        return "%s (%s)" % (self.__class__.__name__, self.db_binary)

    def __iter__(self):
        return self
    
    def __next__(self):
        self.db_iter_idx += 1
        if self.db_iter_idx >= len(self):
            raise StopIteration
        return self[self.db_iter_idx]
    
    def __getitem__(self, index):
        env = self.db_env
        key = self.db_keys[index]
        with env.begin(write=False) as txn:
            byteflow = txn.get(key)
            if byteflow is None: return None, None
        img = Image.open(io.BytesIO(byteflow)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = key.decode('utf-8')
        label = label[:label.rfind('/')]
        label = label2num[label]

        return img, label

if __name__ == "__main__":
    pass