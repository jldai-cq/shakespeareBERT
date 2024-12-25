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
plt.show()