import function as ftn
import numpy as np
import torch
import os
from tqdm import tqdm

# ---------------- GPU 선택 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# -----------------------------------------

for_test = False

# model name
model_name = ''

# just load model and test
model_path = ''

# data load
test_path = ''
test_images, test_cls = ftn.load_image(test_path, 10000, preprocessing = False)

# Build network (pytorch)
model = ftn.RoRNet().to(device)

model.load_state_dict(torch.load(model_path + model_name, map_location=device), strict = False)
model.eval()

print('Model loaded')
print('Testing...')

count_1 = 0

for it in tqdm(range(10000 if not for_test else 1000), ncols=120):
    img = test_images[it:it+1]
    img = np.transpose(img, (0, 3, 1, 2))

    with torch.no_grad():
        pred = model(torch.from_numpy(img.astype(np.float32)).to(device))

        pred_np = pred.detach().cpu().numpy().reshape(10)
        pred_label = int(np.argmax(pred_np))
        gt = int(test_cls[it])
        if gt == pred_label:
            count_1 += 1


acc = count_1 / 10000 * 100 if not for_test else count_1 / 1000 * 100

print('Test Finished...')
print('Path : %s' % (model_path + model_name))
print("Accuracy   : %.4f " % (acc))
