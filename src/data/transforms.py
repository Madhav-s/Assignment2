from torchvision import transforms


class Compose:
    def __init__(self, train: bool = False):
        if train:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
            ])

    def __call__(self, image):
        return self.transforms(image)
