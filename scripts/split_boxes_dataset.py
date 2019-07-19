import os
from os.path import join
import json
import shutil
import numpy as np
from tqdm import tqdm

data_path = '../../Opensource_datasets/Boxes'
json_path = join(data_path, 'Info/train.json')

# load json
js = json.load(open(json_path, 'r'))
images_info = js['image']

# get number info of dataset
num_image = len(images_info)
ids = [x['id'] for x in images_info]
num_ids = len(set(ids))

# choose the 0-2999 id for train and 3000-4018 id for test
train_ids = 3000
folders = ['train', 'test', 'query']
for folder in folders:
    if not os.path.exists(join(data_path, folder)):
        os.makedirs(join(data_path, folder))

# copy train test images
for image_info in tqdm(images_info):
    copy_folder = ''

    if int(image_info['id']) < train_ids:
        copy_folder = 'train'
    else:
        copy_folder = 'test'

    old_path = join(data_path, 'Image', image_info['image_name'])
    copy_path = join(data_path, copy_folder, image_info['image_name'])
    # print('copy image from %s to %s' % (old_path, copy_path))
    shutil.copy(old_path, copy_path)


# # randomly copy an p image to query image
# copy_folder = folders[-1]
# for i in tqdm(range(train_ids, num_ids)):
#     image_ps = [x['image_name'] for x in images_info if x['image_name'].find('%4d_p_' % i) > -1]
#     # print(image_ps)
#     random_choice = np.random.randint(len(image_ps))
#     old_path = join(data_path, 'Image', image_ps[random_choice])
#     copy_path = join(data_path, copy_folder, image_ps[random_choice])
#     # print('copy image from %s to %s' % (old_path, copy_path))
#     shutil.copy(old_path, copy_path)

# As the readme file says, we should set p images as gallery and q images as query
# Remove query folder and move q image in test folder to query folder
shutil.rmtree(join(data_path, 'query'))
os.makedirs(join(data_path, 'query'))
for image in os.listdir(join(data_path, 'test')):
    if image.split('_')[1] == 'g':
        shutil.move(join(data_path, 'test', image), join(data_path, 'query', image))
