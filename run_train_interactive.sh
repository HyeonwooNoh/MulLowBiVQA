# Run train.lua in torch interactive mode.
th -i train.lua \
	-input_img_h5 data/vqa/features.h5 \
	-checkpoint_path model_interactive \
	-batch_size 3 \
	-importance_weighted_training \
	-num_samples 2 \
	-interactive \
   -mhdf5_size 1000
