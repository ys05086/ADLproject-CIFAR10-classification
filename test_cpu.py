# code for model test : using CPU

import function as ftn
import numpy as np
import torch
import os
from tqdm import tqdm

# model name
model_name = ''

# just load model and test
model_path = ''

# data load
test_path = ''
test_images, test_cls = ftn.load_image(test_path, 10000, preprocessing = False)

model = ftn.RoRNet()

model.load_state_dict(torch.load(model_path + model_name)) # , map_location=torch.device('cpu')
model.eval()

print('Model loaded')
print('Testing...')

count = 0

for it in tqdm(range(10000), ncols=120):
    img = test_images[it:it+1]
    img = np.transpose(img, (0, 3, 1, 2))

    with torch.no_grad():
        pred = model(torch.from_numpy(img.astype(np.float32)))

    pred = pred.numpy()
    pred = np.reshape(pred, 10)
    pred = np.argmax(pred)

    gt = test_cls[it]

    if int(gt) == int(pred):
        count += 1

acc = count / 10000 * 100

print("\nTest Finished...")
print("Accuracy   : %.4f " % (acc))
