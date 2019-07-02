from importlib import import_module
from torchvision import transforms
from utils.random_erasing import RandomErasing
from data.sampler import RandomSampler
from torch.utils.data import dataloader
from data.common import Random2DTranslation, ColorAugmentation

class Data:
    def __init__(self, args):
        # 1. Training transforms
        train_list = []

        if args.random_crop:
            train_list.append(Random2DTranslation(args.height, args.width, 0.5))
        else:
            train_list.append(transforms.Resize((args.height, args.width), interpolation=3))

        train_list.append(transforms.RandomHorizontalFlip())

        if args.color_jitter:
            train_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0))

        train_list.append(transforms.ToTensor())
        train_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        if args.random_erasing:
            train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)

        # 2. Test transforms
        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')
            self.train_loader = dataloader.DataLoader(self.trainset,
                                                      sampler=RandomSampler(self.trainset, args.batchid,
                                                                            batch_image=args.batchimage),
                                                      # shuffle=True,
                                                      batch_size=args.batchid * args.batchimage,
                                                      num_workers=args.nThread)
        else:
            self.train_loader = None
        
        if args.data_test in ['Market1501', 'Boxes']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')
        else:
            raise Exception()

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
        