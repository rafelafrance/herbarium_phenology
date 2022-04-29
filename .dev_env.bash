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
python3.10 -m venv .venv
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

pip3 install -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113


# ##############################################################################
# Setup project structure not handled by git. You will want these directories.

mkdir -p data/models/flowering
mkdir -p data/models/fruiting
mkdir -p data/models/leaf_out
mkdir -p data/output


# ##############################################################################
# Dev only pip installs (not required because they're personal preference)

python -m pip install -U tensorboard
python -m pip install -U pynvim
python -m pip install -U 'python-lsp-server[all]'
python -m pip install -U pre-commit pre-commit-hooks
python -m pip install -U autopep8 flake8 isort pylint yapf pydocstyle black
python -m pip install -U jupyter jupyter_nbextensions_configurator ipyparallel
python -m pip install -U jupyter_nbextensions_configurator jupyterlab_code_formatter

python -m pip install -U jupyterlab
python -m pip install -U jupyterlab_code_formatter
python -m pip install -U jupyterlab-drawio
python -m pip install -U jupyterlab-lsp
python -m pip install -U jupyterlab-spellchecker
python -m pip install -U jupyterlab-git
python -m pip install -U aquirdturtle-collapsible-headings
python -m pip install -U nbdime

python -m pip install -U jupyterlab-git==0.36.0

jupyter labextension install jupyterlab_onedarkpro
jupyter server extension enable --py jupyterlab_git
jupyter serverextension enable --py jupyterlab_code_formatter



# ##############################################################################
# I Run pre-commit hooks (optional)

pre-commit install
