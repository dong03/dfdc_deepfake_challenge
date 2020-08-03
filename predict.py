import argparse
import os
import re
import time
import pdb
import json
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from training.zoo.classifiers import DeepFakeClassifier
from torch.backends import cudnn
from training.datasets.classifier_dataset import DeepFakeClassifierDataset, collate_function

from training.tools.utils import AverageMeter, read_annotations, Progbar, predict_set, evaluate
from training.transforms.albu import IsotropicResize

import warnings
warnings.filterwarnings("ignore")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
from albumentations import Compose, PadIfNeeded


def create_val_transforms(size=300):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict test videos")
    arg = parser.add_argument
    arg('--weights-dir', type=str, default="weights", help="path to directory with checkpoints")
    arg('--models', nargs='+', required=True, help="checkpoint files")
    arg('--test_dir', default="", type=str, help="path to directory with videos")
    arg('--gpu', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    modelss = []
    model_paths = [os.path.join(args.weights_dir, model) for model in args.models]
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
        print("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        model.eval()
        model.cuda()
        del checkpoint
        modelss.append([model.half()])
    print("load models finish")
    # test_dirs = [
    #     '/data/dongchengbo/VisualSearch/TIMI/TIMI_test/annotations/TIMI_test.txt',
    #     '/data/dongchengbo/VisualSearch/meso-net/meso-net_val.txt',
    #     '/data/dongchengbo/VisualSearch/wza_mini/wza_mini.txt',
    #     '/data/lixirong/face_tamper_detection/train_test_datasets/DFDC/DFDCdev_lyb/test_imgs_with_label.txt',
    #     '/data/dongchengbo/VisualSearch/dfdc_dfv2_ff++_timit_withface_val.txt']
    test_dirs = ['/data/dongchengbo/VisualSearch/ffbenchmark.txt']
    data_name = [os.path.basename(each).split('.')[0] for each in test_dirs]


    for each in list(zip(test_dirs, data_name)):

        print("begin to pred %s" % each[1])
        annotations = read_annotations(each[0])
        # test_samples = [x.strip() for x in open(args.test_dir).readlines() if x.strip()]
        # annotations = [(x,0) for x in test_samples]

        test_set = DeepFakeClassifierDataset(
            annotations=annotations,
            mode="val",
            balance=False,
            transforms=create_val_transforms(380))

        test_loader = DataLoader(
            dataset=test_set,
            num_workers=8,
            batch_size=128,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_function
        )
        probs_avg = []
        for (index,models) in enumerate(modelss):
            probs, gt_labels, names = predict_set(models,test_loader,{'run_type':'test','debug':0,'data_type':'half'})

            # probs = probs.reshape((-1, 1))
            # gt_labels = gt_labels.reshape(-1)
            # probs_2 = np.hstack([1 - probs, probs])
            # # pdb.set_trace()
            # metrix = evaluate(gt_labels, probs > 0.5, probs)
            # print("model: %s\ndata: %s"%(os.path.basename(model_paths[0]),each[1]))
            # print(metrix)
            # np.save('results/%s_%s_label' % (os.path.basename(model_paths[0]),each[1]), gt_labels)
            # np.save('results/%s_%s_score' % (os.path.basename(model_paths[0]),each[1]), probs)
            # print('\n')

            probs = probs.reshape(-1)
            probs_avg.append(probs)
            results = {}
            label = ['real','fake']
            for i in range(len(names)):
                results[os.path.split(names[i])[-1]] = label[probs[i]>0.5]

            with open('submision/%s.json'%(os.path.basename(model_paths[index])), "w", encoding='utf8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        #pdb.set_trace()
        probs_avg = np.array(probs_avg)
        np.save('submision/merge',probs_avg)
        probs_avg = np.mean(probs_avg,axis=0)
        results = {}
        for i in range(len(names)):
            results[os.path.split(names[i])[-1]] = label[probs[i] > 0.5]
        with open('submision/merge.json', "w", encoding='utf8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)