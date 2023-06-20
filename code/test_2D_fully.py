import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt) # Average Surface Distance
    # hd95 = metric.binary.hd95(pred, gt)
    return dice


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    # fourth_metric = calculate_metric_percase(prediction == 4, label == 4)
    # fifth_metric = calculate_metric_percase(prediction == 5, label == 5)
    # sixth_metric = calculate_metric_percase(prediction == 6, label == 6)
    # seventh_metric = calculate_metric_percase(prediction == 7, label == 7)
    # eighth_metric = calculate_metric_percase(prediction == 8, label == 8)
    # ninth_metric = calculate_metric_percase(prediction == 9, label == 9)
    # tenth_metric = calculate_metric_percase(prediction == 10, label == 10)
    # eleventh_metric = calculate_metric_percase(prediction == 11, label == 11)
    # twelfth_metric = calculate_metric_percase(prediction == 12, label == 12)
    # thirteenth_metric = calculate_metric_percase(prediction == 13, label == 13)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    # return first_metric, second_metric, third_metric, fourth_metric, fifth_metric, sixth_metric, seventh_metric, eighth_metric, ninth_metric, tenth_metric, eleventh_metric, twelfth_metric, thirteenth_metric
    return first_metric, second_metric, third_metric



def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    # snapshot_path = "../checkpoints/newattunet_best_model1.pth"
    # snapshot_path = "../model/oldTrain/ACDC/Cross_Teaching_Between_CNN_Transformer_7/pnet/pnet_best_model1.pth"
    snapshot_path = "../model/oldTrain/ACDC/Cross_Teaching_Between_CNN_Transformer_7/newattunet/newattunet_best_model1.pth"
    # snapshot_path = "../model/oldTrain/ACDC/Cross_Teaching_Between_CNN_Transformer_7/newattr2unet/newattr2unet_best_model1.pth"
    # snapshot_path = "../model/oldTrain/ACDC/Cross_Teaching_Between_CNN_Transformer_7/unet/unet_best_model1.pth"
    # snapshot_path = "../checkpoints/unet_best_model1.pth"
    # net.load_state_dict(torch.load(snapshot_path)["model"])
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    # fourth_total = 0.0
    # fifth_total = 0.0
    # sixth_total = 0.0
    # seventh_total = 0.0
    # eighth_total = 0.0
    # ninth_total = 0.0
    # tenth_total = 0.0
    # eleventh_total = 0.0
    # twelfth_total = 0.0
    # thirteenth_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        # first_metric, second_metric, third_metric, fourth_metric, fifth_metric, sixth_metric, seventh_metric, eighth_metric, ninth_metric, tenth_metric, eleventh_metric, twelfth_metric, thirteenth_metric = test_single_volume(
        #     case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        # fourth_total += np.asarray(fourth_metric)
        # fifth_total += np.asarray(fifth_metric)
        # sixth_total += np.asarray(sixth_metric)
        # seventh_total += np.asarray(seventh_metric)
        # eighth_total += np.asarray(eighth_metric)
        # ninth_total += np.asarray(ninth_metric)
        # tenth_total += np.asarray(tenth_metric)
        # eleventh_total += np.asarray(eleventh_metric)
        # twelfth_total += np.asarray(twelfth_metric)
        # thirteenth_total += np.asarray(thirteenth_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    # avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list), fourth_total / len(image_list), fifth_total / len(image_list), sixth_total / len(image_list), seventh_total / len(image_list), eighth_total / len(image_list), ninth_total / len(image_list), tenth_total / len(image_list), eleventh_total / len(image_list), twelfth_total / len(image_list), thirteenth_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
