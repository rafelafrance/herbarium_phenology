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
# Install a language library for spacy

python -m spacy download en_core_web_sm


# ##############################################################################
# Use the 2nd line if you don't have traiter installed locally

# pip install -e ../traiter/traiter
 pip install git+https://github.com/rafelafrance/traiter.git@master#egg=traiter


# ###############################################################################
# Setup pytorch (Uncomment the one that works for your computer. GPU is better.)
# You will absolutely need one of these.

pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html# pip3 install torch torchvision
# pip3 install torch torchvision


# ##############################################################################
# Setup project structure not handled by git. You will want these directories.

mkdir -p data/models/flowering
mkdir -p data/models/fruiting
mkdir -p data/models/leaf_out
mkdir -p data/output


# ##############################################################################
# Dev only pip installs (not required because they're personal preference)

pip install -U pynvim
pip install -U 'python-lsp-server[all]'
pip install -U pre-commit pre-commit-hooks
pip install -U autopep8 flake8 isort pylint yapf pydocstyle black
pip install -U jupyter jupyter_nbextensions_configurator ipyparallel
pip install -U jupyterlab jupyterlab_code_formatter jupyterlab-drawio
pip install -U jupyterlab-lsp jupyterlab-spellchecker jupyterlab-git
pip install -U aquirdturtle-collapsible-headings
pip install -U nbdime

jupyter labextension install jupyterlab_onedarkpro
jupyter server extension enable --py jupyterlab_git
jupyter serverextension enable --py jupyterlab_code_formatter
