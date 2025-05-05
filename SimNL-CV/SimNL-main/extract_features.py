import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *


def extract_few_shot_feature(cfg, clip_model, train_loader_cache):
    cache_keys = []
    cache_values = []
    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
    torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return


def extract_val_test_feature(cfg, split, clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
    torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
    return


# For positive prompts
def extract_text_feature(cfg, classnames, prompt_path, clip_model, template):
    try:
        f = open(prompt_path)
        prompts = json.load(f)
        with torch.no_grad():
            clip_weights = []
            for i, classname in enumerate(classnames):
                # Tokenize the prompts
                classname = classname.replace('_', ' ')

                template_texts = [t.format(classname) for t in template]
                cupl_texts = prompts.get(classname, template_texts)  # Fallback to template if not found
                texts = template_texts

                texts_token = clip.tokenize(texts, truncate=True).cuda()
                # prompt ensemble for ImageNet
                class_embeddings = clip_model.encode_text(texts_token)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_template.pt")

        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                # Tokenize the prompts
                classname = classname.replace('_', ' ')

                template_texts = [t.format(classname) for t in template]
                cupl_texts = prompts.get(classname, template_texts)  # Fallback to template if not found
                texts = cupl_texts

                texts_token = clip.tokenize(texts, truncate=True).cuda()
                class_embeddings = clip_model.encode_text(texts_token)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
            print(clip_weights.shape)
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_cupl.pt")
    except Exception as e:
        print(f"Error in extract_text_feature: {e}")
        # Create fallback text features using just templates
        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                classname = classname.replace('_', ' ')
                template_texts = [t.format(classname) for t in template]
                texts_token = clip.tokenize(template_texts, truncate=True).cuda()
                class_embeddings = clip_model.encode_text(texts_token)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_template.pt")
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_cupl.pt")
    return


# For negative prompts
def extract_text_feature2(cfg, classnames, prompt_path, clip_model, template):
    try:
        f = open(prompt_path)
        prompts = json.load(f)
        with torch.no_grad():
            clip_weights = []
            for i, classname in enumerate(classnames):
                # Tokenize the prompts
                classname = classname.replace('_', ' ')

                template_texts = [t.format(classname) for t in template]
                cupl_texts = prompts.get(classname, template_texts)  # Fallback to template if not found
                texts = template_texts

                texts_token = clip.tokenize(texts, truncate=True).cuda()
                # prompt ensemble for ImageNet
                class_embeddings = clip_model.encode_text(texts_token)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_negative_template.pt")
        print(clip_weights.shape)
    except Exception as e:
        print(f"Error in extract_text_feature2: {e}")
        # Create fallback text features using just templates
        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                classname = classname.replace('_', ' ')
                template_texts = [t.format(classname) for t in template]
                texts_token = clip.tokenize(template_texts, truncate=True).cuda()
                class_embeddings = clip_model.encode_text(texts_token)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_negative_template.pt")


def process_dataset(dataset_name, data_path, k_shot_list):
    """Process a single dataset with all shot numbers"""

    try:
        print(f"\n{'=' * 50}\nProcessing dataset: {dataset_name}\n{'=' * 50}")

        cfg = yaml.load(open(f'configs/{dataset_name}.yaml', 'r'), Loader=yaml.Loader)

        cache_dir = os.path.join('./caches', cfg['dataset'])
        os.makedirs(cache_dir, exist_ok=True)
        cfg['cache_dir'] = cache_dir

        # First extract features for validation and test sets (only once per dataset)
        random.seed(1)
        torch.manual_seed(1)

        # Build the dataset with a temporary shot value (will be overridden later)
        temp_shot = k_shot_list[0]
        dataset = build_dataset(dataset_name, data_path, temp_shot)

        # Create val and test loaders
        val_loader = build_data_loader(data_source=dataset.val, batch_size=64,
                                       is_train=False, tfm=preprocess, shuffle=False)

        test_loader = build_data_loader(data_source=dataset.test, batch_size=64,
                                        is_train=False, tfm=preprocess, shuffle=False)

        # Extract val/test features (only once per dataset)
        print("\nExtracting validation features...")
        extract_val_test_feature(cfg, "val", clip_model, val_loader)

        print("\nExtracting test features...")
        extract_val_test_feature(cfg, "test", clip_model, test_loader)

        # Extract text features (only once per dataset)
        print("\nExtracting text features...")
        extract_text_feature(cfg, dataset.classnames, dataset.cupl_path, clip_model, dataset.template)
        extract_text_feature2(cfg, dataset.classnames, dataset.cupl_path, clip_model, dataset.negative_template)

        # Then process each shot number
        for k in k_shot_list:
            print(f"\nProcessing {k}-shot features...")
            random.seed(1)
            torch.manual_seed(1)

            cfg['shots'] = k

            # Rebuild the dataset with the current shot value
            dataset = build_dataset(dataset_name, data_path, k)

            # Create training data loader
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))])

            train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256,
                                                   tfm=train_transform, is_train=True, shuffle=False)

            # Extract few-shot features
            print("\nConstructing cache model by few-shot visual features and labels.")
            extract_few_shot_feature(cfg, clip_model, train_loader_cache)

        print(f"Successfully processed dataset: {dataset_name}")
        return True

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Load CLIP model
    clip_model, preprocess = clip.load('RN50')
    clip_model.eval()

    # Specific dataset list with EuroSAT as the default dataset
    all_datasets = ["eurosat", "caltech101", 'dtd', 'fgvc', 'food101',
                    'oxford_flowers', 'oxford_pets', 'ucf101']
    k_shot = [1, 2, 4, 8, 16]

    # Updated data path
    data_path = r'C:\Users\diyad\Downloads\SimNL-main\SimNL-main\DATA'

    # Process each dataset independently
    successful_datasets = []
    failed_datasets = []

    for dataset_name in all_datasets:
        success = process_dataset(dataset_name, data_path, k_shot)
        if success:
            successful_datasets.append(dataset_name)
        else:
            failed_datasets.append(dataset_name)

    # Print summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print(f"\nSuccessfully processed {len(successful_datasets)} datasets:")
    for idx, name in enumerate(successful_datasets):
        print(f"{idx + 1}. {name}")

    if failed_datasets:
        print(f"\nFailed to process {len(failed_datasets)} datasets:")
        for idx, name in enumerate(failed_datasets):
            print(f"{idx + 1}. {name}")
        print("\nPlease check the error messages above for details on failed datasets.")
    else:
        print("\nAll datasets were processed successfully!")