th train.lua \
	-input_img_h5 data/vqa/features.h5 \
	-save_checkpoint_every 5000 \
	-seconds \
	-batch_size 100 \
	-checkpoint_path model_default_seconds/
