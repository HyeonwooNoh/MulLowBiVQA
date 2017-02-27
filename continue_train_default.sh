# Define regular expressions for input argument checking
re_int='^[0-9]+$'

# Check and set load_checkpoint_path
LOAD_CHECKPOINT_PATH=$1
if [ "${LOAD_CHECKPOINT_PATH}" == "" ]; then
	echo "Error: third argument LOAD_CHECKPOINT_PATH is not provided"
	exit 1
fi
echo "load_checkpoint_path: ${LOAD_CHECKPOINT_PATH}"

# Check and set previous_iters
PREVIOUS_ITERS=$2
if ! [[ $PREVIOUS_ITERS =~ $re_int ]] ; then
	echo "Error: fourth argument PREVIOUS_ITERS should be a number"
	exit 1
fi
echo "previous_iters: ${PREVIOUS_ITERS}"

# Set checkpoint_path
CHECKPOINT_PATH="model_default/"
echo "checkpoint_path: ${CHECKPOINT_PATH}"

# Run script
th train.lua \
	-input_img_h5 data/vqa/features.h5 \
	-save_checkpoint_every 5000 \
	-checkpoint_path ${CHECKPOINT_PATH} \
	-load_checkpoint_path ${LOAD_CHECKPOINT_PATH} \
	-previous_iters ${PREVIOUS_ITERS}
