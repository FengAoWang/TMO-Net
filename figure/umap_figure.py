from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import torch
import numpy as np
digits = load_digits()

test_cancer = ['LGG',  'KIRC',  'HNSC']

all_cancer_embedding = []
all_cancer_target = []
for i, cancer in enumerate(test_cancer):
    cancer_embedding = torch.load(f'data_v2/{cancer}_embedding.pt', map_location='cpu')
    target_label = torch.Tensor([i for j in range(cancer_embedding.size(0))])
    all_cancer_embedding.append(cancer_embedding.detach().cpu().numpy())
    all_cancer_target.append(target_label.detach().cpu().numpy())

all_cancer_embedding = np.concatenate(all_cancer_embedding, axis=0)
all_cancer_target = np.concatenate(all_cancer_target, axis=0)

import umap.umap_ as umap
import numpy as np


reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(all_cancer_embedding)
print(embedding)

plt.scatter(embedding[:, 0], embedding[:, 1], c=all_cancer_target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(4)-0.5).set_ticks(np.arange(3))
plt.title('UMAP projection of the Digits dataset')
plt.show()

