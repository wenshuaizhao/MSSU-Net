import os,sys,glob,shutil


ROOT_DIR = os.path.abspath("../")  #This is now the directory of ThreeDUnet
sys.path.append(ROOT_DIR)  # To find local version of the library


data_train='D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/data_train'
data_test='D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/data_test'

imagesTr='D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted\Task11_KITS\imagesTr'
labelsTr='D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted\Task11_KITS\labelsTr'
imagesTs='D:\WenshuaiZhao\ProjectFiles/NNUnet/nnunet/base/nnUNet_raw_splitted\Task11_KITS\imagesTs'

for subject_folder in glob.glob(os.path.join(data_train, "*")):  # Get all the folders in the input folders
    basename=os.path.basename(subject_folder)#case_00000;case_00001...

    name_1 = "imaging.nii.gz"
    name_2="segmentation.nii.gz"
    file_path_1 = os.path.join(subject_folder, name_1)
    file_path_2 = os.path.join(subject_folder, name_2)

    destination_imagesTr=os.path.join(imagesTr,(basename+"_0000.nii.gz"))
    destination_labelsTr = os.path.join(labelsTr, (basename+".nii.gz"))
    shutil.copy2(file_path_1,destination_imagesTr)
    shutil.copy2(file_path_2,destination_labelsTr)
    print('Finished: ',basename)

for subject_folder in glob.glob(os.path.join(data_test, "*")):  # Get all the folders in the input folders
    basename = os.path.basename(subject_folder)  # case_00000;case_00001...

    name_1 = "imaging.nii.gz"
    file_path_1 = os.path.join(subject_folder, name_1)
    destination_imagesTs = os.path.join(imagesTs, (basename+"_0000.nii.gz"))
    shutil.copy2(file_path_1, destination_imagesTs)
    print('Finished: ', basename)




