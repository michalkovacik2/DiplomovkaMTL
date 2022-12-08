from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

'''
[age] nis an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others 
'''

class UTKFace(Dataset):
    def __init__(self, resolution, image_paths):
        # define transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Set inputs and labels
        self.image_paths = image_paths
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []

        for path in image_paths:
            filename = path.split("_")
            if len(filename) == 4:
                self.images.append(f'UTKFace/{path}')
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # load image
        img = Image.open(self.images[index]).convert('RGB')
        # transform
        img = self.transform(img)
        # get labels
        age = self.ages[index]
        gender = self.genders[index]
        eth = self.races[index]

        sample = {'image': img, 'age': age, 'gender': gender, 'ethnicity': eth}
        return sample
