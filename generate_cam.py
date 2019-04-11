import torch
import numpy as np
import os
import misc 
import resnet
import torch.nn as nn

def main():
    batch_size = 32
    img_size = 224 
    save_dir = './plots/' 
    iteration = 1
    train_loader, test_loader, class_num = misc.load_imagenet(batch_size, img_size)

    model = CAM().cuda()

    for idx, data in enumerate(test_loader):
        image, labels = data
        image, labels = image.cuda(), labels.cuda()

        cam, pred = model(image)
        cam =  nn.functional.interpolate(cam, (img_size, img_size), mode='bilinear')

        misc.plot_samples_from_images(image, cam, batch_size, save_dir, 'cam'+str(idx))
        if idx == iteration-1:
            break

class CAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet.resnet18(pretrained=True)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def cam_generator(self, pred, feature):
        ### 1. get top class index
        # pred: (batch, #class)
        # feature: (batch, #inputFM, w, h)
        sorted_logit, sorted_index = torch.sort(torch.squeeze(pred.detach()), dim=1,  descending=True)
        sorted_prob = sorted_logit[:, 0]
        #sorted_prob = self.softmax(sorted_logit[:, 0])
        # sorted_index: (batch, 1)

        selected_weight = torch.cat([torch.index_select(self.model.fc.weight, 0, idx).unsqueeze(0)
                for idx in  sorted_index[:, 0]])
        # weight: (#class, #inputFM)
        # selected_weight: (batch, 1, #inputFM) 

        s = feature.size()
        cams = torch.bmm(selected_weight, feature.view(s[0], s[1], s[2]*s[3]))
        # cams = (batch, 1, w, h)
        return cams.view(s[0], 1, s[2], s[3])

    def forward(self, img):
        with torch.no_grad():
            pred, feature = self.model(img)
            cam = self.cam_generator(pred, feature)
        return cam, pred


if __name__ == "__main__":
    main()
