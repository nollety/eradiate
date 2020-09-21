#!/bin/bash
eval "$(conda shell.bash hook)"
ROOT=`pwd`
CONDA_ENV_NAME="eradiate"

echo "Creating conda env ${CONDA_ENV_NAME} ..."
conda env create --force --file ../deps/requirements_conda.yml --name ${CONDA_ENV_NAME} || { echo "${CONDA_ENV_NAME} env creation failed" ; exit 1; }

echo "Updating conda env ${CONDA_ENV_NAME} with dev packages ..."
conda env update --quiet --file ../deps/requirements_dev_conda.yml --name ${CONDA_ENV_NAME} || { echo "${CONDA_ENV_NAME} env update failed" ; exit 1; }

echo "Updating conda env ${CONDA_ENV_NAME} with jupyter lab ..."
conda env update --quiet --file ../deps/requirements_jupyter_conda.yml --name ${CONDA_ENV_NAME} || { echo "${CONDA_ENV_NAME} env update failed" ; exit 1; }

echo "Copying environment variable setup scripts ..."
conda activate ${CONDA_ENV_NAME}

SCRIPT="#!/bin/sh"
SCRIPT="${SCRIPT}\nexport ERADIATE_DIR=${ROOT}"
SCRIPT="${SCRIPT}\nexport MITSUBA_DIR=\"\$ERADIATE_DIR/ext/mitsuba2\""
SCRIPT="${SCRIPT}\nexport PYTHONPATH=\"\$ERADIATE_DIR/build/dist/python:\$PYTHONPATH\""
SCRIPT="${SCRIPT}\nexport PATH=\"\$ERADIATE_DIR/build/dist:\$PATH\""
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
echo -e ${SCRIPT} > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh

SCRIPT="#!/bin/sh"
# Clean up PATH
SCRIPT="${SCRIPT}\nmatch=\"\$ERADIATE_DIR/build/dist:\""
SCRIPT="${SCRIPT}\nexport PATH=\${PATH//\$match/}"
# Clean up PYTHONPATH
SCRIPT="${SCRIPT}\nmatch=\"\$ERADIATE_DIR/build/dist/python:\""
SCRIPT="${SCRIPT}\nexport PYTHONPATH=\"\${PYTHONPATH//\$match/}\""
# Remove other environment variables
SCRIPT="${SCRIPT}\nunset ERADIATE_DIR"
SCRIPT="${SCRIPT}\nunset MITSUBA_DIR"
mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
echo -e ${SCRIPT} > ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh

# Enable ipywidgets support
echo "Enabling ipywidgets within jupyter lab ..."
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Install Eradiate in dev mode
echo "Installing Eradiate to conda env ${CONDA_ENV_NAME} in developer mode ..."
python setup.py develop > /dev/null

conda deactivate

echo
echo "Activate conda env ${CONDA_ENV_NAME} using the following command:"
echo
echo "    conda activate ${CONDA_ENV_NAME}"
echo