import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
from pathlib import Path
from torchvision import transforms
import net
import Residual_easy as re
from torch.utils.data.sampler import BatchSampler
import net

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def train_transform():
    transform_list = [
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(128),
            transforms.ToTensor()
            ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


#parameter design

#data path
content_dir = "/D/reflection/COCO2017/train2017/"
reflect_dir = "/D/reflection/reflection_style_transfer/reflection"

#Discriminator
n_epochs = 10
learn_R = 2e-5
b1 = 0.5
b2 = 0.999
batch_size = 2
n_threads = 1


content_tf = train_transform()
reflect_tf = train_transform()

content_dataset = FlatFolderDataset(content_dir, content_tf)
reflect_dataset = FlatFolderDataset(reflect_dir, reflect_tf)



content_iter = iter(data.DataLoader(
    content_dataset, batch_size=batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=n_threads))
reflect_iter = iter(data.DataLoader(
    reflect_dataset, batch_size=batch_size,
    sampler=InfiniteSamplerWrapper(reflect_dataset),
    num_workers=n_threads))


adversarial_loss = torch.nn.MSELoss()
device = torch.device('cuda')



#G = Generator()
z = torch.randn(2,3,128,128)
print(z.view(z.size(0),-1))
res = re.Residual( 512 , "bn" , "relu" )
vgg = net.vgg
decoder = net.decoder

teacher = net.Net( vgg , decoder , res )
student = net.Net( vgg , decoder , res )

teacher.train()
teacher.to(device)

student.train()
student.to(device)

optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=learn_R, betas=(b1, b2))
optimizer_student = torch.optim.Adam(student.parameters(), lr=learn_R, betas=(b1, b2))

for epoch in range(n_epochs):

    #REAL NOISE 資料讀取

    content_images = next(content_iter).to(device)
    reflect_images = next(reflect_iter).to(device)

    print(content_images.size())

    optimizer_teacher.zero_grad()

    # Generate a batch of images
    # This Generator change to the Adain(Net)
    gen_imgs,_,loss_re = teacher(reflect_images,content_images)

#    print("G　:" + str(gen_imgs.size()))
#    print("R  :" + str(reflect_images.size()))
    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(gen_imgs, reflect_images)
    g_loss.backward()
    optimizer_teacher.step()




    optimizer_student.zero_grad()

    student_imgs,loss_p,_ = student(content_images,content_images)

    str_loss = adversarial_loss(student_imgs, reflect_images)
    print("teacher_loss :" + str(g_loss))
    print("str_loss :" + str(str_loss))
    str_loss += loss_p
    str_loss.backward()
    optimizer_student.step()
    





    #  Train Discriminator

#    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
#    real_loss = adversarial_loss(discriminator(real_imgs), valid)
#    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

    #這部分還可以做別的調整
#    d_loss = (real_loss + fake_loss) / 2

#    d_loss.backward()
#    optimizer_D.step()






