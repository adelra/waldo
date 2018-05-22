<<<<<<< HEAD
<<<<<<< HEAD
=======
#!/usr/bin/env python3

>>>>>>> waldo-seg/master
=======
#!/usr/bin/env python3

>>>>>>> waldo-seg/master
import torch
import argparse
import os
import sys
<<<<<<< HEAD
<<<<<<< HEAD
import torchvision
import random
from torchvision import transforms as tsf
from models.Unet import UNet
from dataset import Dataset_dsb2018
from waldo.segmenter import ObjectSegmenter
=======
=======
>>>>>>> waldo-seg/master
import random
import numpy as np
from models.Unet import UNet
from train import sample
from dataset import Dataset_dsb2018
from waldo.segmenter import ObjectSegmenter
from waldo.core_config import CoreConfig
from waldo.data_visualization import visualize_mask
from unet_config import UnetConfig
<<<<<<< HEAD
>>>>>>> waldo-seg/master
=======
>>>>>>> waldo-seg/master


parser = argparse.ArgumentParser(description='Pytorch DSB2018 setup')
parser.add_argument('model', type=str,
                    help='path to final model')
<<<<<<< HEAD
<<<<<<< HEAD
parser.add_argument('--img-height', default=128, type=int,
                    help='Height of resized images')
parser.add_argument('--img-width', default=128, type=int,
                    help='width of resized images')
parser.add_argument('--img-channels', default=3, type=int,
                    help='Number of channels of images')
parser.add_argument('--name', default='Unet-5', type=str,
                    help='name of experiment')
parser.add_argument('--val-data', default='./data/val.pth.tar', type=str,
                    help='Path of processed validation data')
parser.add_argument('--test-data', default='./data/test.pth.tar', type=str,
                    help='Path of processed test data')
parser.add_argument('--num-classes', default=2, type=int,
                    help='Number of classes to classify')
parser.add_argument('--num-offsets', default=10, type=int,
                    help='Number of points in offset list')

=======
=======
>>>>>>> waldo-seg/master
parser.add_argument('--dir', default='exp/unet', type=str,
                    help='directory to store segmentation results')
parser.add_argument('--train-dir', default='./data/val.pth.tar', type=str,
                    help='Path of processed validation data')
parser.add_argument('--train-image-size', default=128, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).'
                    'These are derived from the input images'
                    ' by padding and then random cropping.')
parser.add_argument('--core-config', default='', type=str,
                    help='path of core configuration file')
parser.add_argument('--unet-config', default='', type=str,
                    help='path of network configuration file')
<<<<<<< HEAD
>>>>>>> waldo-seg/master
=======
>>>>>>> waldo-seg/master
random.seed(0)


def main():
    global args
    args = parser.parse_args()
<<<<<<< HEAD
<<<<<<< HEAD
    args.batch_size = 1
    args.depth = 16

    # # of classes, # of offsets
    model = UNet(args.num_classes, args.num_offsets)
=======
=======
>>>>>>> waldo-seg/master
    args.batch_size = 1  # only segment one image for experiment

    # loading core configuration
    c_config = CoreConfig()
    if args.core_config == '':
        print('No core config file given, using default core configuration')
    if not os.path.exists(args.core_config):
        sys.exit('Cannot find the config file: {}'.format(args.core_config))
    else:
        c_config.read(args.core_config)
        print('Using core configuration from {}'.format(args.core_config))

    # loading Unet configuration
    u_config = UnetConfig()
    if args.unet_config == '':
        print('No unet config file given, using default unet configuration')
    if not os.path.exists(args.unet_config):
        sys.exit('Cannot find the unet configuration file: {}'.format(
            args.unet_config))
    else:
        # need c_config for validation reason
        u_config.read(args.unet_config, args.train_image_size)
        print('Using unet configuration from {}'.format(args.unet_config))

    offset_list = c_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    num_classes = c_config.num_classes
    num_colors = c_config.num_colors
    num_offsets = len(c_config.offsets)
    # model configurations from unet config
    start_filters = u_config.start_filters
    up_mode = u_config.up_mode
    merge_mode = u_config.merge_mode
    depth = u_config.depth

    model = UNet(num_classes, num_offsets,
                 in_channels=num_colors, depth=depth,
                 start_filts=start_filters,
                 up_mode=up_mode,
                 merge_mode=merge_mode)
<<<<<<< HEAD
>>>>>>> waldo-seg/master
=======
>>>>>>> waldo-seg/master

    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
<<<<<<< HEAD
<<<<<<< HEAD
        model.cpu()
        offset_list = checkpoint['offset_list']
        print("loaded.")
        print("offsets are {}".format(offset_list))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    s_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((args.img_height, args.img_width)),
        tsf.ToTensor(),
    ])

    testset = Dataset_dsb2018(args.val_data, s_trans, offset_list,
                              args.num_classes, args.img_height, args.img_width)
=======
=======
>>>>>>> waldo-seg/master
        print("loaded.")
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    model.eval()  # convert the model into evaluation mode

    val_data = args.train_dir + '/' + 'val.pth.tar'

    testset = Dataset_dsb2018(val_data, c_config, args.train_image_size)
<<<<<<< HEAD
>>>>>>> waldo-seg/master
=======
>>>>>>> waldo-seg/master
    print('Total samples in the test set: {0}'.format(len(testset)))

    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

<<<<<<< HEAD
<<<<<<< HEAD
    data_iter = iter(dataloader)
    # data_iter.next()
    img, class_id, sameness = data_iter.next()
    torch.set_printoptions(threshold=5000)
    torchvision.utils.save_image(img, 'input.png')
    torchvision.utils.save_image(sameness[0, 0, :, :], 'sameness0.png')
    torchvision.utils.save_image(sameness[0, 1, :, :], 'sameness1.png')
    torchvision.utils.save_image(
        class_id[0, 0, :, :], 'class0.png')  # backgrnd
    torchvision.utils.save_image(class_id[0, 1, :, :], 'class1.png')  # cells

    model.eval()  # convert the model into evaluation mode

    img = torch.autograd.Variable(img)
    predictions = model(img)
    predictions = predictions.data
    # [batch-idx, class-idx, row, col]
    class_pred = predictions[0, :args.num_classes, :, :]
    # [batch-idx, offset-idx, row, col]
    adj_pred = predictions[0, args.num_classes:, :, :]

    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            adj_pred[i, :, :], 'sameness_pred{}.png'.format(i))
    for i in range(args.num_classes):
        torchvision.utils.save_image(
            class_pred[i, :, :], 'class_pred{}.png'.format(i))

    seg = ObjectSegmenter(class_pred.numpy(),
                          adj_pred.numpy(), args.num_classes, offset_list)
#    seg = ObjectSegmenter(class_id[0, :, :, :].numpy(), sameness[0, :, :, :].numpy(), args.num_classes, offset_list)
    seg.run_segmentation()

    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            adj_pred[i, :, :], 'sameness_pred{}.png'.format(i))
    for i in range(args.num_classes):
        torchvision.utils.save_image(
            class_pred[i, :, :], 'class_pred{}.png'.format(i))
=======
=======
>>>>>>> waldo-seg/master
    seg_dir = '{}/seg'.format(args.dir)
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    img, class_pred, adj_pred = sample(model, dataloader, seg_dir, c_config)

    seg = ObjectSegmenter(class_pred[0].detach().numpy(),
                          adj_pred[0].detach().numpy()[:2, :, :],
                          num_classes, offset_list[:2], seg_dir)
    mask_pred, object_class = seg.run_segmentation()
    x = {}
    x['img'] = np.moveaxis(img[0].numpy(), 0, -1)
    x['mask'] = mask_pred.astype(int)
    x['object_class'] = object_class
    visualize_mask(x, c_config)
<<<<<<< HEAD
>>>>>>> waldo-seg/master
=======
>>>>>>> waldo-seg/master


if __name__ == '__main__':
    main()
