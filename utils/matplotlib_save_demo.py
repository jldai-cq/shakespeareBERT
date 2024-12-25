import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

train_loss =[1,2,3,4]
eval_loss = [0.2, 0.3, 0.1, 0.05]
steps = range(1, len(train_loss) + 1)

plt.plot(steps, train_loss, label='Train Loss')
plt.plot(steps, eval_loss, label='Eval Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()

# 自动保存图像到文件而非显示出来
plt.savefig('./data/loss_curve.png')  # 设置保存路径和文件名
plt.close()  # 关闭图像，避免图像在下一次绘制时重叠

plt.show()