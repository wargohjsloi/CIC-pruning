import torch
from torch.nn.functional import cosine_similarity

# 加载.pth文件并获取存储的张量列表
file_paths = ['./outputs/cifar_moco_many.pth', './outputs/cifar_moco_mid.pth', './outputs/cifar_moco_few.pth']
tensors = []
for file_path in file_paths:
    idxs = torch.load(file_path)
    # print(idxs.type())
    # assert 0
    idx = torch.tensor(idxs[0], dtype=torch.float)
    for ii, ts in enumerate(idxs[1:]):
        idx = torch.cat((idx, torch.tensor(ts, dtype=torch.float) + ii * 10000), dim=0)
    # print(idx)
    tensors.append(idx)

# assert 0

# 计算相似度
similarities = []
for i in range(len(tensors)):
    for j in range(i+1, len(tensors)):
        set1 = set(tensors[i].tolist())
        set2 = set(tensors[j].tolist())

        # 计算交集和并集大小
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        # 计算Jaccard相似度
        jaccard_similarity = intersection / union

        # similarity = cosine_similarity(tensors[i], tensors[j], dim=0)
        similarities.append(jaccard_similarity)

# 打印相似度结果
for idx, similarity in enumerate(similarities):
    print(f'Similarity between file{idx+1} and file{idx+2}: {similarity}')
