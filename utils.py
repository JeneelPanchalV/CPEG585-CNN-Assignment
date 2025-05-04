import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory."""
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


def read_json(fpath):
    """Read json file from a path."""
    try:
        with open(fpath, 'r') as f:
            obj = json.load(f)
        return obj
    except FileNotFoundError:
        print(f"Error: The file '{fpath}' was not found.")
        print("Please check that:\n"
              "1. You have downloaded the dataset\n"
              "2. Your config file has the correct 'root_path'\n"
              "3. The dataset includes the required split file\n"
              "4. The path structure matches what the code expects")
        raise


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using PIL.Image."""
    if not osp.exists(path):
        print(f"[Missing Image] {path}")
        # Return dummy image (black RGB image)
        return Image.new('RGB', (224, 224), color=(0, 0, 0))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(f"Cannot read image from {path}, retrying...")


class Datum:
    """Data instance structure."""
    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self): return self._impath
    @property
    def label(self): return self._label
    @property
    def domain(self): return self._domain
    @property
    def classname(self): return self._classname


class DatasetBase:
    dataset_dir = ''
    domains = []

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x
        self._train_u = train_u
        self._val = val
        self._test = test
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self): return self._train_x
    @property
    def train_u(self): return self._train_u
    @property
    def val(self): return self._val
    @property
    def test(self): return self._test
    @property
    def lab2cname(self): return self._lab2cname
    @property
    def classnames(self): return self._classnames
    @property
    def num_classes(self): return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = sorted(mapping.keys())
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    f"Input domain must belong to {self.domains}, but got [{domain}]"
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))
        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError
        print("Extracting file ...")
        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            with zipfile.ZipFile(dst, 'r') as zip_ref:
                zip_ref.extractall(osp.dirname(dst))
        print(f"File extracted to {osp.dirname(dst)}")

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        if num_shots < 1:
            return data_sources[0] if len(data_sources) == 1 else data_sources
        print(f"Creating a {num_shots}-shot dataset")
        output = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled = random.sample(items, num_shots)
                else:
                    sampled = random.choices(items, k=num_shots) if repeat else items
                dataset.extend(sampled)
            output.append(dataset)
        return output[0] if len(output) == 1 else output

    def split_dataset_by_label(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output

    def split_dataset_by_domain(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.domain].append(item)
        return output


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False,
                 return_img0=False, k_tfm=1):
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(f"Cannot augment image {self.k_tfm} times because transform is None")

        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = [
            T.Resize(input_size, interpolation=interp_mode),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        output = {'label': item.label, 'domain': item.domain, 'impath': item.impath}
        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = f'img{i+1}' if i > 0 else 'img'
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = [tfm(img0) for _ in range(self.k_tfm)]
        return img_list[0] if len(img_list) == 1 else img_list


def build_data_loader(data_source=None, batch_size=64, input_size=224,
                      tfm=None, is_train=True, shuffle=False, dataset_wrapper=None):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    num_workers = 0 if os.name == 'nt' else 8  # safer for Windows

    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )

    assert len(data_loader) > 0
    return data_loader
