# 将lent导出为wts
import torch
import numpy
import struct
wts_path = "artifacts/lenet.wts"
model = torch.load('artifacts/lenet.pt',weights_only=False)
# model.to("cpu").eval()

print(model)

'''
10
fc1 10 3fexfg 23asae123


'''

with open(wts_path,"w") as f:
    f.write(f"{len(model.keys())}\n")
    for key,item in model.items():
        item_flatten = item.reshape(-1).cpu().numpy()
        f.write(f"{key} {len(item_flatten)}")
        for ii in item_flatten:
            f.write(" ")
            f.write(struct.pack('>f',float(ii)).hex())
        f.write("\n")
    
