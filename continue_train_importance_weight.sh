# Define regular expressions for input argument checking
re_int='^[0-9]+$'
re_float='^[0-9]+([.][0-9]+)?$'

# Check and set num_samples
NUM_SAMPLES=$1
if ! [[ $NUM_SAMPLES =~ $re_int ]] ; then
	echo "Error: first argument NUM_SAMPLES should be a number"
	exit 1
fi
echo "num_samples: ${NUM_SAMPLES}"

# Check and set dropout ratio
DROPOUT=$2
if ! [[ $DROPOUT =~ $re_float ]] ; then
	echo "Error: second argument DROPOUT should be a floating point number"
	exit 1
fi
echo "dropout: ${DROPOUT}"

# Check and set load_checkpoint_path
LOAD_CHECKPOINT_PATH=$3
if [ "${LOAD_CHECKPOINT_PATH}" == "" ]; then
	echo "Error: third argument LOAD_CHECKPOINT_PATH is not provided"
	exit 1
fi
echo "load_checkpoint_path: ${LOAD_CHECKPOINT_PATH}"

# Check and set previous_iters
PREVIOUS_ITERS=$4
if ! [[ $PREVIOUS_ITERS =~ $re_int ]] ; then
	echo "Error: fourth argument PREVIOUS_ITERS should be a number"
	exit 1
fi
echo "previous_iters: ${PREVIOUS_ITERS}"

# Set checkpoint_path
CHECKPOINT_PATH_BASE=model_importance_weight
CHECKPOINT_PATH="${CHECKPOINT_PATH_BASE}/dropout_${DROPOUT}_num_samples_${NUM_SAMPLES}/"
echo "checkpoint_path: ${CHECKPOINT_PATH}"

# Run script
th train.lua \
	-input_img_h5 data/vqa/features.h5 \
	-save_checkpoint_every 5000 \
	-checkpoint_path ${CHECKPOINT_PATH} \
	-importance_weighted_training \
	-num_samples ${NUM_SAMPLES} \
	-dropout ${DROPOUT} \
	-load_checkpoint_path ${LOAD_CHECKPOINT_PATH} \
	-previous_iters ${PREVIOUS_ITERS}
