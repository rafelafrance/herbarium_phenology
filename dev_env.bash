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

python -m spacy download en_core_web_sm

pip install -e ../traiter/traiter

# ##############################################################################
# Dev only pip installs (not required b/c they're personal preference)

pip install -U pynvim
pip install -U 'python-lsp-server[all]'
pip install -U pre-commit pre-commit-hooks
pip install -U autopep8 flake8 isort pylint yapf pydocstyle black
pip install -U jupyter jupyter_nbextensions_configurator ipyparallel
pip install -U jupyter_nbextensions_configurator jupyterlab_code_formatter

# ###############################################################################
# Setup pytorch (Uncomment the one that works for your computer. GPU is better.)
# You will absolutely need one of these.

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# ##############################################################################
# Setup project structure not handled by git. You will want these directories.

mkdir -p data/models
mkdir -p data/output
