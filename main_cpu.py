# i made the condition similar to the paper
# iter is too much: spent 10 hours for training 100000 iteration
# and batch_size =/= mini_batch size
# btw paper's contidtion is this:
#                                 Data Preprocessing  : subtract mean, devide std
#                                 Data Augmentation   : flipping
#                                 optimizer           : Adam
#                                 weight_decay = 1e-4 : L2 regularization
#                                 momentum = 0.9      : to add acceleration
# but, Data Preprocessing is actually not needed      : used dividing by 255.0
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import function as ftn

# user set param
# num_training = 200000 # 200000
learning_rate = 0.1
model_save_path = './Project1_1_alter/model/128-omega/'
batch_size = 128
restore_iter = 64000
num_training = (50000 // batch_size) * 500 # 500 epochs with almost full data

restore_lr = 0.001
brestore = True

# Load Data
train_path = 'Project1_1/CIFAR10/train/'
test_path = 'Project1_1/CIFAR10/test/'
train_images, train_cls = ftn.load_image(train_path, 50000, preprocessing = False)
test_images, test_cls = ftn.load_image(test_path, 1000, preprocessing = False)

# bulid network
model = ftn.RoRNet()

# restore data
if brestore == True:
    print('Model Restore')
    model.load_state_dict(torch.load(model_save_path + 'model_%d.pt' % restore_iter))

# loss ftn and param settings
optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, lr=restore_lr if brestore else learning_rate) # same as RoR paper
loss = torch.nn.CrossEntropyLoss()

model_ckpt = [] # iteration, acc, model_path

# training Network
for it in tqdm(range(restore_iter if brestore else 0, num_training), ncols=120, desc='Training Progress'):
    # mini batch
    batch_img, batch_cls = ftn.mini_batch_training(train_images, train_cls, batch_size)

    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_img.astype(np.float32))) # (B,C,H,W)
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long)

    train_loss = loss(pred, cls_tensor)
    train_loss.backward()
    optimizer.step()

    if it % 1000 == 0 and it > (restore_iter if brestore else 0): # 0 is for testing saving option
        tqdm.write("\niteration: %d " % it)
        tqdm.write("train loss : %f " % train_loss.item())
        tqdm.write('Evaluating the Model...')
        model.eval()

        count = 0
        for itest in tqdm(range(1000), ncols=100):
            test_img = test_images[itest:itest + 1]
            test_img = np.transpose(test_img, (0, 3, 1, 2)) # Change to [B, C, H, W]

            with torch.no_grad():
                pred = model(torch.from_numpy(test_img.astype(np.float32)))

            pred = pred.numpy() # [1, 10]
            pred = np.reshape(pred, 10)
            pred = np.argmax(pred)

            gt = test_cls[itest]

            if int(gt) == int(pred):
                count += 1

        acc = count / 1000 * 100

        tqdm.write("Accuracy   : %f " % acc)
        tqdm.write("Current LR : %f " % optimizer.param_groups[0]['lr'])

        tqdm.write('SAVING MODEL')
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        torch.save(model.state_dict(), model_save_path + 'model_%d.pt' % it)
        tqdm.write('MODEL SAVED')

        # check point data save
        model_ckpt.append((it, acc, model_save_path + 'model_%d.pt' % it))
    
    # # more easy to read: not one line
    if it == 20000:
        optimizer.param_groups[0]['lr'] = 0.01

    if it == 40000:
        optimizer.param_groups[0]['lr'] = 0.001 # end is near here

    if it == 60000:
        optimizer.param_groups[0]['lr'] = 0.0001

    # if it == 100000:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1

    # if it == 150000:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1

print('TRAINING DONE')
print('BEST MODEL')
model_ckpt = sorted(model_ckpt, key=lambda x: x[1], reverse=True)
print(model_ckpt[0])
