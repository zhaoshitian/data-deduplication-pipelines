from autofaiss import build_index
import numpy as np
from tqdm import tqdm
import faiss
import os
import json

def cluster_embeddings_with_faiss(embeddings, n_clusters=10, niter=20, use_gpu=True):
    """
    使用 Faiss 对embedding向量进行K-means聚类。

    参数：
    - embeddings: ndarray, 大小为 (num_samples, num_features)，表示待聚类的embedding向量。
    - n_clusters: int, 聚类的簇数。
    - niter: int, 迭代次数。
    - use_gpu: bool, 是否使用GPU加速。

    返回：
    - labels: ndarray, 每个向量所属的簇标签。
    - centers: ndarray, 每个簇的中心点。
    """
    # 确保输入是float32类型
    embeddings = embeddings.astype('float32')

    # 初始化 KMeans
    kmeans = faiss.Kmeans(d=embeddings.shape[1], k=n_clusters, niter=niter, verbose=True, gpu=use_gpu)

    # 进行聚类
    kmeans.train(embeddings)

    # 获取每个向量所属的簇
    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.flatten()

    # 获取每个簇的中心
    centers = kmeans.centroids

    return labels, centers

def find_duplicate_items(vecs, items, index, thresh):
    dups = set()
    for i in tqdm(range(len(vecs))):
        qs = vecs[i]  # Current vector
        qid = items[i]  # Current item ID
        lims, D, I = index.range_search(np.expand_dims(qs, axis=0), thresh)
        if qid in dups:
            continue
        start = lims[0]
        end = lims[1]
        duplicate_indices = I[start:end]
        duplicate_ids = []
        for j in duplicate_indices:
            if items[j] != qid:
                duplicate_ids.append(items[j])
        dups.update(duplicate_ids)
    return dups

def load_npy_file(file_path):
    """
    尝试加载 .npy 文件，如果遇到 EOFError 则捕获异常并跳过该文件。
    
    参数:
    - file_path: .npy 文件的路径
    
    返回:
    - 成功加载的 numpy 数组或 None（如果加载失败）
    """
    try:
        data = np.load(file_path)
        return data
    except EOFError:
        print(f"文件 {file_path} 读取失败，文件可能已损坏或为空，跳过此文件。")
        return None
    except Exception as e:
        # 捕获其他潜在的异常
        print(f"加载 {file_path} 时遇到错误: {e}")
        return None

def collect_embeds(root_dir):
    embeds_list = []
    image_name_list = []
    for sub_dir in tqdm(os.listdir(root_dir)):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for data_name in os.listdir(sub_dir_path):#:[:300]:
                #print('collecting:',data_name)
                data_path = os.path.join(sub_dir_path, data_name)
                item_embed = load_npy_file(data_path)
                if item_embed is None:
                    print(f"Skipping: {data_path}, file is empty or None or destroyed")
                    continue
                elif item_embed.shape != (1024,):
                    print(f"Skipping: {data_path}, shape: {item_embed.shape}")
                else:
                    embeds_list.append(item_embed)
                    image_name_list.append(f"part_{sub_dir.split('_', -1)[-1]}_name_{data_name.split('.', -1)[0]}")
    with open("/mnt/petrelfs/gaopeng/swj/sscd/file_names.json", "w") as json_file:
       json.dump(image_name_list, json_file, indent=4)
    embeds_list = np.stack(embeds_list)
    return embeds_list, image_name_list


if __name__ == "__main__":
    # embeddings = np.load("/mnt/petrelfs/zhaoshitian/data/embeds_laion_27M/embeds_part0.npy")[:1000]
    root_dir = "/mnt/petrelfs/gaopeng/swj/sscd/data/Synthesized_data/embeddings/laion_24M"
    embeddings, image_name_list = collect_embeds(root_dir)
    print(f"shape of embedding: {embeddings.shape}")
    np.save("/mnt/petrelfs/gaopeng/swj/sscd/data/Synthesized_data/embeddings/laion_24M/all_embeds.npy", embeddings)
    # labels, centers = cluster_embeddings_with_faiss(embeddings, n_clusters=10, niter=20, use_gpu=False)
    # print(f"shape of embeddings: {embeddings.shape}")
    # print(f"labels: {labels}")
    # print(f"centers: {centers.shape}")

    index, index_infos = build_index(embeddings, save_on_disk=False)

    items = [j for j in range(len(embeddings))]
    dups = find_duplicate_items(embeddings, image_name_list, index, 0.5)

    print(dups)