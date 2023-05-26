import torch
from torch.autograd import Variable

from model.LPRNET import LPRNet
from model.STN import STNet
import struct

model_path = './weights/Final_STN_model.pth'

model = STNet()
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

image = torch.ones(1, 3, 24, 94)
if torch.cuda.is_available():
    image = image.cuda()

model.eval()
print(model)
print('image shape ', image.shape)
preds = model(image)

f = open("STNet.wts", 'w')
f.write("{}\n".format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")
