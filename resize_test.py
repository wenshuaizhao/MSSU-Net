import os,sys
ROOT_DIR=os.path.abspath("../")
sys.path.append(ROOT_DIR)
# print(ROOT_DIR)
import SimpleITK as sitk
import numpy as np
from nnunet.preprocessing.preprocessing import resample_data_or_seg
import torch
import nibabel as nib
import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
image_path='/cache/WenshuaiZhao/ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted/Task11_KITS/imagesTr/case_00001_0000.nii.gz'
label_path='/cache/WenshuaiZhao/ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted/Task11_KITS/labelsTr/case_00001.nii.gz'
new_shape=tuple([50,200,200])
image=nib.load(image_path)
label=nib.load(label_path)
image_data=image.get_data()
image_data=image_data[np.newaxis, np.newaxis,:,:,:]
print('image data after expanding dimension is:',image_data.shape)
image_data_torch=torch.from_numpy(image_data).float()
label_data=label.get_data()
label_data=label_data[np.newaxis, np.newaxis,:,:,:]
label_data_torch=torch.from_numpy(label_data).float()


print('Original shape is :',image_data.shape)
print('image_data_torch shape is:',image_data_torch.shape)
image_affine=image.get_affine()
# image_data_new=resize_segmentation(image_data,new_shape)
# label_data_new=resize_segmentation(label_data,new_shape)
label_data_new_torch=torch.nn.functional.interpolate(label_data_torch,size=new_shape,mode='trilinear',align_corners=True)
print('torch unique after trilinear is:',torch.unique(label_data_new_torch))
unique_list=list(torch.unique(label_data_new_torch))
print('length is:',len(unique_list))

label_data_new_torch_nearest=torch.nn.functional.interpolate(label_data_torch,size=new_shape,mode='nearest')
print('torch unique after nearest is:',torch.unique(label_data_new_torch_nearest))
unique_list_1=list(torch.unique(label_data_new_torch_nearest))
print('length is:',len(unique_list_1))

# img=nib.Nifti1Image(image_data_new,image_affine)
# lab=nib.Nifti1Image(label_data_new,image_affine)
lab_torch_tri=nib.Nifti1Image(label_data_new_torch,image_affine)
lab_torch_near=nib.Nifti1Image(label_data_new_torch_nearest,image_affine)
image_new_path='/cache/WenshuaiZhao/ProjectFiles/NNUnet/nnunet/new_image.nii.gz'
label_new_path='/cache/WenshuaiZhao/ProjectFiles/NNUnet/nnunet/new_label.nii.gz'
label_new_path_tri='/cache/WenshuaiZhao/ProjectFiles/NNUnet/nnunet/new_label_tri.nii.gz'
label_new_path_near='/cache/WenshuaiZhao/ProjectFiles/NNUnet/nnunet/new_label_near.nii.gz'
# img.to_filename(image_new_path)
# lab.to_filename(label_new_path)
lab_torch_tri.to_filename(label_new_path_tri)
lab_torch_near.to_filename(label_new_path_near)



# image=sitk.ReadImage(image_path)
# image_data=sitk.GetArrayFromImage(image)
# image_data_1=np.expand_dims(sitk.GetArrayFromImage(image),axis=0)
# print(image_data_1.shape)
# #(602, 602, 101)
# new_shape=(300,300,50)
# image_data_new=resample_data_or_seg(image_data_1,new_shape,is_seg=False)
# print(image_data_new.shape)
# out=sitk.GetImageFromArray(image_data_new)


# sitk.WriteImage(out,image_new_path)


