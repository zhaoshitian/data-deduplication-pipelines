from torchvision import transforms
import torch
from PIL import Image
from data_reader import read_general
import json
import os
from tqdm import tqdm
import torch
import datetime
import numpy as np

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])
skew_320 = transforms.Compose([
    transforms.Resize([320, 320]),
    transforms.ToTensor(),
    normalize,
])


def get_embed(model, image_url):
    image_bytes_io = read_general(image_url)
    img = Image.open(image_bytes_io).convert('RGB')
    batch = small_288(img).unsqueeze(0).cuda()
    embedding = model(batch)[0, :]
    return embedding


def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="gen_internvl")
    parser.add_argument("--part_index", type=int, default=0)
    args = parser.parse_args()

    model = torch.jit.load("/mnt/petrelfs/gaopeng/swj/sscd/sscd_disc_large.torchscript.pt").cuda()

    save_root_path = "/mnt/petrelfs/gaopeng/swj/sscd/data/Synthesized_data/embeddings/laion_24M"
    json_file_path = "/mnt/hwfile/alpha_vl/gaopeng/share_data_bak/data_filter/collect_annotations/data/laion_coarse_24M.json"
    item_list = json.load(open(json_file_path, "r"))[args.part_index*100000: (1+args.part_index)*100000]

    embed_list = []

    for i, item in tqdm(enumerate(item_list), total=len(item_list)):

        try:
            part_path = os.path.join(save_root_path, f"part_{args.part_index}")
            checkpath(part_path)
            
            item_name = f"part_{args.part_index}/{item['sample_id']}.npy"
            data_save_file_path = os.path.join(save_root_path, item_name)
            if os.path.exists(data_save_file_path):
                continue
            image_url = item['image_url']
            embed = get_embed(model, image_url).cpu().detach().numpy()
            np.save(data_save_file_path, embed)
            # embed_list.append(embed)
            # print(f"shape of embed: {embed.shape}")

            # print(f"item: {i + args.part_index * 100000}")
        except:
            print("something wrong.")

        # part_index = (i + args.part_index * 100000) // 100000
        # part_path = os.path.join(save_root_path, f"part_{part_index}")
        # checkpath(part_path)
    # embed_list = np.stack(embed_list)
    # np.save(f"/mnt/petrelfs/zhaoshitian/data/embeds_laion_27M/embeds_part{args.part_index}.npy", embed_list)
        