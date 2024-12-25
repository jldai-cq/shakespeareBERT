import json
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

with open('./model/shakespeareBert_v3/checkpoint-11110/trainer_state.json', 'r', )as f:
        log_history = json.load(f)

log_history = log_history['log_history']
# 绘制损失函数曲线图
# 初始化两个空列表来存储 loss 和 eval_loss 的值
train_loss = []
train_epoch = []
eval_loss = []
eval_epoch = []

# 遍历 log_history 列表
for entry in log_history:
    # 如果字典中有 'loss' 键，则将其值添加到 train_loss 列表中
    if 'loss' in entry:
        train_loss.append(entry['loss'])
        if 'epoch' in entry:
            train_epoch.append(entry['epoch'])
    # 如果字典中有 'eval_loss' 键，则将其值添加到 eval_loss 列表中
    if 'eval_loss' in entry:
        eval_loss.append(entry['eval_loss'])
        if 'epoch' in entry:
            eval_epoch.append(entry['epoch'])


plt.plot(train_epoch, train_loss, label='train_loss')
plt.plot(eval_epoch[:-1], eval_loss[:-1], label='eval_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.show()
# 自动保存图像到文件而非显示出来
plt.savefig('./model/shakespeareBert_v3/loss_curve.png')
plt.close()