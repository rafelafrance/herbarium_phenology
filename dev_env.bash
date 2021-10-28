#!/usr/bin/env bash

# #################################################################################
# Setup the virtual environment for development.
# You may need to "pip install --user virtualenv" globally.
# This is not required but some form of project isolation (conda virtual env etc.)
# is strongly encouraged.

if [[ ! -z "$VIRTUAL_ENV" ]]; then
  echo "'deactivate' before running this script."
  exit 1
fi

rm -rf .venv
virtualenv -p python3.9 .venv
source ./.venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi


# ##############################################################################
# Dev only pip installs (not required b/c they're personal preference)

pip install -U pynvim
pip install -U 'python-lsp-server[all]'
pip install -U pre-commit pre-commit-hooks
pip install -U autopep8 isort pylint yapf pydocstyle black
pip install -U jupyter_nbextensions_configurator jupyterlab_code_formatter
pip install -U ipyparallel


# ###############################################################################
# Setup pytorch (Uncomment the one that works for your computer. GPU is better.)
# You will absolutely need one of these.

# pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


# ##############################################################################
# Setup project structure not handled by git. You will want these directories.

mkdir -p data
mkdir -p models
mkdir -p output
