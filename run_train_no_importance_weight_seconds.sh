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

# Set checkpoint_path
CHECKPOINT_PATH_BASE=model_no_importance_weight_seconds
CHECKPOINT_PATH="${CHECKPOINT_PATH_BASE}/dropout_${DROPOUT}_num_samples_${NUM_SAMPLES}/"
LOG_PATH="${CHECKPOINT_PATH}/training_log.txt"
echo "checkpoint_path: ${CHECKPOINT_PATH}"

# Run script
th train.lua \
	-input_img_h5 data/vqa_ssd/features.h5 \
	-save_checkpoint_every 5000 \
	-checkpoint_path ${CHECKPOINT_PATH} \
	-importance_weighted_training \
	-num_samples ${NUM_SAMPLES} \
	-dropout ${DROPOUT} \
	-seconds \
	-batch_size 100 \
	| tee ${LOG_PATH}
