# Check whether model_path is provided.
MODEL_PATH=$1
if [ "${MODEL_PATH}" == "" ]; then
	echo "Provide model_path as a first argument."
	exit 1
fi
echo "model_path: ${MODEL_PATH}"

# Check whether out_path is provided.
OUT_PATH=$2
if [ "${OUT_PATH}" == "" ]; then
	echo "Provide out_path as a second argument."
	exit 1
fi
echo "out_path: ${OUT_PATH}"

# Check whether backend is correctly provided.
BACKEND=$3
if [ "${BACKEND}" == "" ]; then
	BACKEND="cudnn"
fi
if [ "${BACKEND}" != "nn" ] && [ "${BACKEND}" != "cudnn" ]; then
	echo "Wrong backend augment is provided: ${BACKEND}"
	exit 1
fi
echo "backend: ${BACKEND}"

# Other fixed augments
INPUT_IMG_H5="data/vqa/test_features.h5"
echo "input_img_h5: ${INPUT_IMG_H5}"

# Run script
th eval.lua \
	-model_path "${MODEL_PATH}" \
	-out_path "${OUT_PATH}" \
	-input_img_h5 "${INPUT_IMG_H5}" \
	-backend "${BACKEND}"
