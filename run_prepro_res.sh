IMAGE_ROOT=/media/hyeonwoonoh/PetaDrive/Projects/cumulus/2015_SUMMER/002_image_qa/002_hyeonwoonoh/Image-Question-Answering/data/MSCOCO/images/
CNN_MODEL=/media/hyeonwoonoh/PetaDrive/Projects/cumulus/2017_SPRING/fb.resnet.torch/pretrained/resnet-152.t7
OUT_PATH=data/vqa/features.h5
th prepro_res.lua -image_root $IMAGE_ROOT -cnn_model $CNN_MODEL -out_path $OUT_PATH
