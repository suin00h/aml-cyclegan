import os
import sys
import caffe
import argparse
import numpy as np
from PIL import Image
from util import *
from cityscapes import cityscapes

parser = argparse.ArgumentParser()
parser.add_argument("--cityscapes_dir", type=str, required=True, help="Path to the original cityscapes dataset")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--gt_dir", type=str, required=True, help="Path to the GT images to be evaluated")
parser.add_argument("--output_dir", type=str, required=True, help="Where to save the evaluation results")
parser.add_argument("--caffemodel_dir", type=str, default='../caffemodel/', help="Where the FCN-8s caffemodel is stored")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu id to use")
parser.add_argument("--split", type=str, default='val', help="Data split to be evaluated")
parser.add_argument("--save_output_images", type=int, default=0, help="Whether to save the FCN output images")
args = parser.parse_args()

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_output_images > 0:
        output_image_dir = os.path.join(args.output_dir, 'image_outputs')
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)

    CS = cityscapes(args.cityscapes_dir)
    n_cl = len(CS.classes)
    label_frames = CS.list_label_frames(args.split)
    # caffe.set_mode_cpu()
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(
        os.path.join(args.caffemodel_dir, 'deploy.prototxt'),
        caffe.TEST,
        weights=os.path.join(args.caffemodel_dir, 'fcn-8s-cityscapes.caffemodel')
    )
    
    hist_label_perframe = np.zeros((n_cl, n_cl))
    hist_perframe = np.zeros((n_cl, n_cl))
    acc = np.zeros((n_cl, n_cl))

    for i, idx in enumerate(label_frames):
        if i % 10 == 0:
            print('Evaluating: %d/%d' % (i, len(label_frames)))

        city = idx.split('_')[0]
        label = CS.load_label(args.split, city, idx)

        im_file = os.path.join(args.result_dir, idx + '_leftImg8bit.png')
        im = np.array(Image.open(im_file).convert('RGB'))

        labelim_file = os.path.join(args.result_dir, idx + '_gtFine.png')
        label_im = np.array(Image.open(labelim_file))
        label_im = CS.rgb_to_trainId(label_im)

        GT_labelim_file = os.path.join(args.gt_dir, idx + '_gtFine.png')
        GT_label_im = np.array(Image.open(GT_labelim_file))
        GT_label_im = CS.rgb_to_trainId(GT_label_im)


        # Resize using PIL to match label shape (W, H)
        resized_im = Image.fromarray(im).resize((label.shape[2], label.shape[1]), Image.BILINEAR)
        im = np.array(resized_im)

        # resized_label_im = np.array(Image.fromarray(label_im).resize((label.shape[2], label.shape[1]), Image.NEAREST))
        # resized_GT_label_im = np.array(Image.fromarray(GT_label_im).resize((label.shape[2], label.shape[1]), Image.NEAREST))

        # out = segrun(net, CS.preprocess(im))
        # hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)
        
        hist_label_perframe += fast_hist(resized_GT_label_im.flatten(), resized_label_im.flatten(), n_cl)

        acc += per_pixel_accuracy(np.array(Image.open(labelim_file)), np.array(Image.open(GT_labelim_file)))

        if args.save_output_images > 0:
            label_im = Image.fromarray(CS.palette(label))
            pred_im = Image.fromarray(CS.palette(out))
            input_im = Image.fromarray(im)
            # resized_label_im_I = Image.fromarray(CS.palette(resized_label_im).astype(np.uint8))
            # resized_GT_label_im_I = Image.fromarray(CS.palette(resized_GT_label_im).astype(np.uint8))

            label_im.save(os.path.join(output_image_dir, f'{i}_gt.jpg'))
            pred_im.save(os.path.join(output_image_dir, f'{i}_pred.jpg'))
            input_im.save(os.path.join(output_image_dir, f'{i}_input.jpg'))
            # resized_label_im_I.save(os.path.join(output_image_dir, f'{i}_input1.jpg'))
            # resized_GT_label_im_I.save(os.path.join(output_image_dir, f'{i}_input2.jpg'))

    # Save evaluation results
    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    mean_pixel_acc_l, mean_class_acc_l, mean_class_iou_l, per_class_acc_l, per_class_iou_l = get_scores(hist_label_perframe)

    # mean_pixel_acc_l = acc.mean()
       

    with open(os.path.join(args.output_dir, 'evaluation_results_l2p.txt'), 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            cl = cl.ljust(15)
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))

    with open(os.path.join(args.output_dir, 'evaluation_results_p2l.txt'), 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc_l)
        f.write('Mean class accuracy: %f\n' % mean_class_acc_l)
        f.write('Mean class IoU: %f\n' % mean_class_iou_l)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            cl = cl.ljust(15)
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc_l[i], per_class_iou_l[i]))

main()
