import function as ftn
import torch
import numpy as np
import os

# user set param
num_training = 100000
learning_rate = 0.1
model_save_path = './model/RESNET-18/'
brestore = True
restore_iter = 1000
batch_size = 32

# data load
train_path = './CIFAR10/train'
test_path = './CIFAR10/test'
train_images, train_cls = ftn.load_image(train_path, 50000) # training 60000 in one time takes too much
test_images, test_cls = ftn.load_image(test_path, 1000)


# Build network (pytorch)
model = ftn.CNN_RESNET_18()


# Model Load?
if brestore == True:
    print('Model restore')
    model.load_state_dict(torch.load(model_save_path + 'model_%d.pt' % restore_iter))
    model.eval()


# loss ftn and param settings
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss = torch.nn.CrossEntropyLoss()

# model temporary save list (for deleting and loading best model)
model_ckpt = [] # (iteration, acc, model_path)

model_ckpt_hof = [] # (iteration, acc, model_path) (hall of fame for best model)

# lr change를 위한 counting
lr_count = 0

# training Network
for it in range(restore_iter, num_training + 1):
    # learning rate control, which we call decay
    # if it >= 20000 and it < 30000:
    #     optimizer.param_groups[0]['lr'] = 0.01
    # elif it >= 30000:
    #     optimizer.param_groups[0]['lr'] = 0.001

    batch_img, batch_cls = ftn.Mini_batch_training(train_images, train_cls, batch_size)
    # batch_img shape : [B, H, W] - > [B, C, H, W]

    # training step
    # iter 중단한데서 시작할 경우, train을 진행하지 않는 방향으로.
    # 중단. lr 문제같음. 따라서 lr을 자동적으로 낮춰주는 조건문 만들기.
    # good. 예상이 맞았음. 갑작스럽게 과도한 LR을 가져와서 문제가 생김.
    if restore_iter == 1000:
        optimizer.param_groups[0]['lr'] = 0.001

    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)))
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long)

    train_loss = loss(pred, cls_tensor)
    train_loss.backward()
    optimizer.step()

    # 추가기능 : 100 step 마다 최고의 모델을 찾아서 로딩한 이후, 그 모델을 불러온 후 학습을 재시작
    # array에 acc를 저장해두고, 1000step 마다 acc를 비교하여 최고 acc 모델을 저장.
    # array 형태를 (iteration, acc, model_path) 로 저장 후 삭제하기 용이하게.

    if it % 1000 == 0 and it > 0:
        print("\niteration  : %d " % it)
        print("train loss : %f " % train_loss.item())
        model.eval()
        count_1 = 0
        for itest in range(1000):
            test_img = test_images[itest:itest + 1] / 255.0
            test_img = np.transpose(test_img, (0, 3, 1, 2)) # Change to [B, C, H, W]

            with torch.no_grad():
                pred = model(torch.from_numpy(test_img.astype(np.float32)))

            pred = pred.numpy() # [1, 10]
            pred = np.reshape(pred, 10)
            pred = np.argmax(pred)

            gt = test_cls[itest]

            if int(gt) == int(pred):
                count_1 += 1

        acc = count_1 / 1000 * 100

        if acc > 55 and acc <= 79:
            if optimizer.param_groups[0]['lr'] > 0.01:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10 # 0.01
            else:
                pass
        elif acc > 79:
            if optimizer.param_groups[0]['lr'] > 0.001:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10 # 0.001
            else:
                pass

        print("Accuracy   : %.4f " % (acc))
        print("Current LR : ", optimizer.param_groups[0]['lr'])


        print('SAVING MODEL')
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        torch.save(model.state_dict(), model_save_path + 'model_%d.pt' % it)
        print('MODEL SAVED')
        
        # 모델 데이터 저장
        model_ckpt.append((it, count_1 / 1000 * 100, model_save_path + 'model_%d.pt' % it))

    if it % 10000 == 0:
        if it == 0:
            model_ckpt = []
            continue

        # (iteration, acc, model_path) 10000 step 마다 최고의 모델을 저장
        # 이때 1000 iter 앞뒤로 모델을 비교해서 더 나아졌다면 해당 모델을 로딩, 아니라면 현 모델 유지.

        # acc 기준 모델 정렬기 using sorted() 함수 sorted(리스트, key, reverse = boolean )
        model_ckpt = sorted(model_ckpt, key =lambda x: x[1], reverse = False) # acc 기준 오름차순 정렬

        # 최고 acc 모델 로딩
        print('Loading The Best Model among the last 100 * 10 iterations')
        print('Best Accuracy : %.4f' % model_ckpt[-1][1])

        # 이전 모델과 비교
        if it == 10000:
            model_ckpt_hof.append(model_ckpt[-1]) # hof에 추가
        else:
            if model_ckpt[-1][1] > model_ckpt_hof[-1][1]:
                model_ckpt_hof.append(model_ckpt[-1]) # hof에 추가 - 자동으로 오름차순 정렬이 됨.
                print('HOF Data Updated!')

                # # learning rate 조절 - 모델의 성능이 향상되었을때만. - 최적화?
                # # ACC 높아졌을때만. (40퍼센트 이상.)
                # if model_ckpt[-1][1] > 40:
                #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2 # 처음 learning rate의 절반으로.
                #     print('Learning Rate Decreased to %.4f' % (optimizer.param_groups[0]['lr']))

                # lr_count 초기화
                lr_count = 0
            else:
                print('HOF Data Unchanged!')
                print('Current Best Accuracy : %.4f' % model_ckpt_hof[-1][1])
                lr_count += 1
                if lr_count >= 2: # 2번 연속으로 lr 감소가 없는 경우, lr 감소 # 처음 5로 설정했을때, 너무 오랜시간을 학습하는 것으로 보여, 2회로 줄임.
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
                    print('Learning Rate Decreased to %.4f' % (optimizer.param_groups[0]['lr']))
                    lr_count = 0

        best_model_path = model_ckpt_hof[-1][2] # iteration, acc, model_path 순서로 저장됨.
        model.load_state_dict(torch.load(best_model_path))
        print('Model Loaded...')

        
        # 필요없는 모델 삭제
        for i in range(len(model_ckpt) - 1):
            if os.path.isfile(model_ckpt[i][2]):
                os.remove(model_ckpt[i][2]) # os.remove(파일경로) : 파일 삭제
        print('Deleted the other models except the best model')

        # model_ckpt 초기화
        model_ckpt = []

# 최고의 모델을 제외한 모델 삭제
for items in range(len(model_ckpt_hof) - 1):
    os.remove(model_ckpt_hof[items][2])
print('Deleted all the other models except the best model')
print('The Best Model : %s with Accuracy %.4f' % (model_ckpt_hof[-1][2], model_ckpt_hof[-1][1]))
