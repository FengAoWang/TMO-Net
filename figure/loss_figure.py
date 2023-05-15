import numpy as np
import matplotlib.pyplot as plt
import torch
import random
# data = open('infer_loss_0.txt', 'r')
# lines = data.readlines()
# loss_array = []
# for line in lines:
#     loss_array.append(float(line))
# print(loss_array)

loss = torch.load('all_pancancer_pretrain_train_loss_fold0.pt', map_location='cpu')

test_loss = torch.load('all_pancancer_pretrain_test_loss_fold0.pt', map_location='cpu')

total_loss = loss[:, 0]
mse_loss = loss[:, 3]
self_encoders_loss = loss[:, 1]
cross_encoders_loss = loss[:, 2]
dsc_loss = loss[:, 4]
test_dsc_loss = test_loss[:, 4]
print(dsc_loss)

# total_loss = np.load('total_loss.npy')
# mse_loss = np.load('mse_loss.npy')
# self_encoders_loss = np.load('self_encoders_loss.npy')
# cross_encoders_loss = np.load('cross_encoders_loss.npy')

epochs = [i for i in range(50)]
# print(epochs)
# loss_array = torch.load('dual_vae_loss.ph')

# plt.plot(epochs, total_loss, 'r-')
# plt.xlabel(u'epochs')
# plt.ylabel(u'loss')
# plt.title('ct infer genomic features loss')
# plt.show()

fig, ax = plt.subplots() # 创建图实例

# ax.plot(epochs, total_loss, label='Total Loss') # 作y1 = x 图，并标记此线名为linear
#
# ax.plot(epochs, self_encoders_loss, label='Self-ELBO Loss') # 作y3 = x^3 图，并标记此线名为cubic
#
# ax.plot(epochs, cross_encoders_loss, label='Cross-ELBO Loss')
#
# ax.plot(epochs, mse_loss, label='Cross-infer MSE Loss') #作y2 = x^2 图，并标记此线名为quadratic

ax.plot(epochs, dsc_loss, label='Train Adversarial loss')
ax.plot(epochs, test_dsc_loss, label='Test Adversarial loss')

ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.set_title('Pretrain loss')
ax.legend()
plt.show()
