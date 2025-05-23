import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets


template = ['a centered satellite photo of {}.',
            'A top-down view of {} arranged in a pattern.']

negative_template = ['a centered satellite photo without {}.',
                     'A top-down view without {} arranged in a pattern.']


NEW_CNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSAT(DatasetBase):

    dataset_dir = 'eurosat'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        
        self.template = template
        self.negative_template = negative_template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_eurosat.json'

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)
    
    def update_classname(self, dataset_old, NEW_CLASSNAMES=None):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(
                impath=item_old.impath,
                label=item_old.label,
                classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
