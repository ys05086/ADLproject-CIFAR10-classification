# 목표기능 
# 1. Data Augmentation을 이용한 학습. - Cut mix, Shuffle ... etc - data에 변화를 줘서 정확도를 끌어올리는 방식.
# 2. Data Shuffle을 통한 overfitting의 방지. - same as 1.
# 3. Early Stopping을 혹시나 적용할 수 있을지. but in this case, not needed. 

import function as ftn
import torch
import numpy as np
import os

# user set param
num_training = 100000
learning_rate = 0.1
model_save_path = './model/'
brestore = False
restore_iter = 1000

# data load
train_path = './CIFAR10/train'
test_path = './CIFAR10/test'
train_images, train_cls = ftn.load_image(train_path, 50000) # training 60000 in one time takes too much
test_images, test_cls = ftn.load_image(test_path, 1000)


# Build network (pytorch)
model = ftn.CNN_v3()


# Model Load?
if brestore == True:
    print('Model restore')
    model.load_state_dict(torch.load(model_save_path + 'model_%d.pt' % restore_iter))
    model.eval()


# loss ftn and param settings
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()

# training Network
for it in range(restore_iter, num_training + 1):
    # learning rate control, which we call decay
    if it >= 10000 and it < 40000:
        optimizer.param_groups[0]['lr'] = 0.01
    elif it >= 40000:
        optimizer.param_groups[0]['lr'] = 0.001

    batch_img, batch_cls = ftn.Mini_batch_training(train_images, train_cls, 64)
    # batch_img shape : [B, H, W] - > [B, C, H, W]

    # training step
    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)))
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long)

    train_loss = loss(pred, cls_tensor)
    train_loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        print("\niteration  : %d " % it)
        print("train loss : %f " % train_loss.item())
        model.eval()
        count = 0
        for itest in range(1000):
            test_img = test_images[itest:itest + 1] # image 그대로.
            test_img = np.transpose(test_img, (0, 3, 1, 2)) # Change to [B, C, H, W]

            with torch.no_grad():
                pred = model(torch.from_numpy(test_img.astype(np.float32)))

            pred = pred.numpy() # [1, 10]
            pred = np.reshape(pred, 10)
            pred = np.argmax(pred)

            gt = test_cls[itest]

            if int(gt) == int(pred):
                count += 1

        print("Accuracy   : %.4f " % (count / 1000 * 100))

        print('SAVING MODEL')
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)

        torch.save(model.state_dict(), model_save_path + 'model_%d.pt' % it)
        print('MODEL SAVED')
