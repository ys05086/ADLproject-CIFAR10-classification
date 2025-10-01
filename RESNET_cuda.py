import function as ftn
import torch
import numpy as np
import os
from tqdm import tqdm

# ---------------- GPU 선택 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# -----------------------------------------

# user set param
num_training = 100000
learning_rate = 0.1
model_save_path = './model/RESNET-NEW-GPU/'
brestore = False
restore_iter = 1000
batch_size = 16

# data load (numpy로 유지)
train_path = './CIFAR10/train'
test_path = './CIFAR10/test'
train_images, train_cls = ftn.load_image(train_path, 50000)
test_images, test_cls = ftn.load_image(test_path, 1000)

# Build network (pytorch)
model = ftn.CNN_RESNET_NEW_Light().to(device)

# Model Load?
if brestore == True:
    print('Model restore')
    state = torch.load(model_save_path + 'model_%d.pt' % restore_iter, map_location=device)
    model.load_state_dict(state)
    model.eval()

# loss ftn and param settings
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss().to(device)

# model temporary save list (for deleting and loading best model)
model_ckpt = []  # (iteration, acc, model_path)

model_ckpt_hof = [] # (iteration, acc, model_path) (hall of fame for best model)

lr_count = 0

# training Network
for it in tqdm(range(num_training + 1), ncols=120):
    # mini-batch (numpy)
    batch_img, batch_cls = ftn.Mini_batch_training(train_images, train_cls, batch_size)

    # numpy -> torch on device
    x = torch.from_numpy(batch_img.astype(np.float32)).to(device)      # (B,C,H,W)
    y = torch.tensor(batch_cls, dtype=torch.long, device=device)       # (B,)

    # training step
    model.train()
    optimizer.zero_grad()
    pred = model(x)
    train_loss = loss(pred, y)
    train_loss.backward()
    optimizer.step()

    # 매 1000 step마다 평가/저장/기록
    if it % 1000 == 0 and it > 0:
        tqdm.write("\niteration  : %d " % it)
        tqdm.write("train loss : %f " % train_loss.item())
        tqdm.write('Evaluating the Model...')
        model.eval()
        count_1 = 0
        with torch.no_grad():
            for itest in tqdm(range(1000), ncols=100):
                test_img = test_images[itest:itest + 1] / 255.0                         # (1,H,W,C)
                test_img = np.transpose(test_img, (0, 3, 1, 2)).astype(np.float32)      # (1,C,H,W)
                tx = torch.from_numpy(test_img).to(device)
                logits = model(tx)
                # CPU로 내려서 numpy 변환 (필요시)
                pred_np = logits.detach().cpu().numpy().reshape(10)
                pred_label = int(np.argmax(pred_np))
                gt = int(test_cls[itest])
                if gt == pred_label:
                    count_1 += 1

        acc1000 = count_1 / 1000 * 100

        if acc1000 > 60 and acc1000 <= 80:
            if optimizer.param_groups[0]['lr'] > 0.01:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            else:
                pass
        elif acc1000 > 80:
            if optimizer.param_groups[0]['lr'] > 0.001:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            else:
                pass

        tqdm.write("Accuracy   : %.4f " % acc1000)
        tqdm.write("Current LR : %.4f " % optimizer.param_groups[0]['lr'])

        tqdm.write('SAVING MODEL')
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        ckpt_path = model_save_path + 'model_%d.pt' % it
        torch.save(model.state_dict(), ckpt_path)
        tqdm.write('MODEL SAVED')

        # 모델 데이터 저장
        model_ckpt.append((it, acc1000, ckpt_path))

    # 매 10000 step마다 최근 10개 중 최고만 로드, 나머지 삭제
    if it % 10000 == 0:
        if it == 0:
            model_ckpt = []
            continue

        model_ckpt = sorted(model_ckpt, key=lambda x: x[1], reverse=False)  # acc asc
        tqdm.write('Loading The Best Model among the last 100 * 10 iterations')
        tqdm.write('Best Accuracy : %.4f' % model_ckpt[-1][1])

                # 이전 모델과 비교
        if it == 10000:
            model_ckpt_hof.append(model_ckpt[-1]) # hof에 추가
        else:
            if model_ckpt[-1][1] > model_ckpt_hof[-1][1]:
                model_ckpt_hof.append(model_ckpt[-1]) # hof에 추가
                tqdm.write('HOF Data Updated!')
                # learning rate 조절 - 모델의 성능이 향상되었을때만. - 최적화
                # ACC 높아졌을때만. (40퍼센트 이상.)
                # if model_ckpt[-1][1] > 40:
                #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2 # 처음 learning rate의 절반으로.
                #     print('Learning Rate Decreased to %.4f' % (optimizer.param_groups[0]['lr']))
                lr_count = 0
            else:
                tqdm.write('HOF Data Unchanged!')
                tqdm.write('Current Best Accuracy : %.4f' % model_ckpt_hof[-1][1])
                lr_count += 1
                if lr_count >= 5: # 5번 연속으로 lr 감소가 없는 경우, lr 감소
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
                    tqdm.write('Learning Rate Decreased to %.4f' % (optimizer.param_groups[0]['lr']))
                    lr_count = 0


        best_model_path = model_ckpt[-1][2]
        # GPU에 맞게 로드
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        tqdm.write('Model Loaded...')

        # 필요없는 모델 삭제
        for i in range(len(model_ckpt) - 1):
            if os.path.isfile(model_ckpt[i][2]):
                os.remove(model_ckpt[i][2])
        tqdm.write('Deleted the other models except the best model')

        # model_ckpt 초기화
        model_ckpt = []


# 최고의 모델을 제외한 모델 삭제
for items in range(len(model_ckpt_hof) - 1):
    os.remove(model_ckpt_hof[items][2])
print('Deleted all the other models except the best model')
print('The Best Model : %s with Accuracy %.4f' % (model_ckpt_hof[-1][2], model_ckpt_hof[-1][1]))
