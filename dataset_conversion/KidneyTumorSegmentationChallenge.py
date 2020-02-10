#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import label


def export_segmentations(indir, outdir):
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        sitk.WriteImage(img, outfname)


def export_segmentations_postprocess(indir, outdir):
    maybe_mkdir_p(outdir)
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        print("\n", n)
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        img_npy = sitk.GetArrayFromImage(img)
        lmap, num_objects = label((img_npy > 0).astype(int))
        sizes = []
        for o in range(1, num_objects + 1):
            sizes.append((lmap == o).sum())
        mx = np.argmax(sizes) + 1
        print(sizes)
        img_npy[lmap != mx] = 0
        img_new = sitk.GetImageFromArray(img_npy)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":

    def load_save_train(args):
        data_file = args
        pat_id = data_file.split("/")[-1]
        # pat_id = "train_" + pat_id.split("-")[-1][:-4]

        # img_itk = sitk.ReadImage(data_file)
        # sitk.WriteImage(img_itk, join(img_dir, pat_id + "_0000.nii.gz"))
        #
        # img_itk = sitk.ReadImage(seg_file)
        # sitk.WriteImage(img_itk, join(lab_dir, pat_id + ".nii.gz"))
        return pat_id

    def load_save_test(args):
        data_file = args
        pat_id = data_file.split("/")[-1]
        # pat_id = "test_" + pat_id.split("-")[-1][:-4]

        # img_itk = sitk.ReadImage(data_file)
        # sitk.WriteImage(img_itk, join(img_dir_te, pat_id + "_0000.nii.gz"))
        return pat_id


    train_image_dir = "D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted\Task11_KITS\imagesTr"
    train_label_dir="D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted\Task11_KITS\labelsTr"
    test_dir = "D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted\Task11_KITS\imagesTs"

    nii_files_tr_data = subfiles(train_image_dir, True, "case", "nii.gz", True)
    print(nii_files_tr_data)
    nii_files_tr_seg = subfiles(train_label_dir, True, "case", "nii.gz", True)

    nii_files_ts = subfiles(test_dir, True, "case", "nii.gz", True)

    # p = Pool(8)
    # train_ids = load_save_train( nii_files_tr_data)
    # test_ids = load_save_test( nii_files_ts)
    # p.close()
    # p.join()

    # output_folder = "/media/fabian/My Book/MedicalDecathlon/MedicalDecathlon_raw_splitted/Task29_LITS"
    output_folder='D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted/Task11_KITS'
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")


    json_dict = OrderedDict()
    json_dict['name'] = "KITS"
    json_dict['description'] = "KITS"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "f off, this is private"
    json_dict['licence'] = "touch it and you die"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "kidney",
        "2": "tumor"
    }
    json_dict['training']=list()
    if len(nii_files_tr_data)==len(nii_files_tr_seg):
        for i in range(len(nii_files_tr_data)):
            s_1 = str(os.path.splitext(os.path.splitext(os.path.basename(nii_files_tr_data[i]))[0])[0][:-5])
            # s_1=str(os.path.splitext(os.path.basename(nii_files_tr_data[i])))
            s_2 = str(os.path.splitext(os.path.splitext(os.path.basename(nii_files_tr_seg[i]))[0])[0])
            # s_2 = str(os.path.splitext(os.path.basename(nii_files_tr_seg[i])))
            dict={'image': "./imagesTr/%s.nii.gz" % s_1, "label": "./labelsTr/%s.nii.gz" % s_2}
            json_dict['training'].append(dict)

    json_dict['test']=list()

    for i in range(len(nii_files_ts)):
        s_1 = str(os.path.splitext(os.path.splitext(os.path.basename(nii_files_ts[i]))[0])[0])

        dict = ["./imagesTs/%s.nii.gz" % s_1]
        json_dict['test'].append(dict)

    json_dict['numTraining'] = 168
    json_dict['numTest'] = 42
    # json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
    #                          train_ids]
    # json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)