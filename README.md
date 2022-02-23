# Herbarium phenology![Python application](https://github.com/rafelafrance/herbarium_phenology/workflows/CI/badge.svg)

Extracting phenological information from digitized herbarium specimens.

## Project Summary

Over the last few decades there has been a lot of effort to digitize herbarium specimens by photographing them and recording any of their annotations into databases, however, this effort has mostly been manual and labor-intensive resulting in only a fraction of herbarium specimens being fully annotated.

This project uses neural networks to automate the annotation of one set of biologically significant traits, relating to [phenology](https://en.wikipedia.org/wiki/Phenology): flowering, fruiting, and leaf-out.

The basic steps are:

1. Obtain a database of herbarium images with corresponding annotations.
   1. We are using the [iDigBio database](https://www.idigbio.org/) for this.
2. Clean and filter the iDigBio database to contain only angiosperm records with images.
3. Find a subset of these records for training that meet the following criteria:
   1. It has an annotation of the presence or absence of at least one of the phenological traits. We use the [spaCy](https://spacy.io/) library to mine the database's free text fields for these traits.
   2. It has an annotation of the specimen's phylogenetic order.
   3. It has exactly one image associated with the specimen. More than one image creates confusion as to which image contains the trait.
4. Train a neural network(s) to recognize the traits. We are using the [pytorch](https://pytorch.org/) library to build the neural networks.
5. Use the networks to annotate records.

This project extends Brian Stucky's work located [here](https://gitlab.com/stuckyb/herbarium_phenology).

## Setup

**TODO**: This is a bit complicated. Make an install script.

1. Create a virtual environment
   1. Make sure you have a virtual environment manager installed. I use `virtualenv`.
      1. `pip install --user virtualenv`
   2. Check out a tag.
      1. `cd /path/to/herbarim_phenology`
      2. `git checkout v0.1` (or another tag)
   3. Create a virtual environment.
      1.`virtualenv -p python3.9 .venv` (You may use python 3.9+)
      2. `source ./.venv/bin/activate`
   4. Install module requirements.
      1. `python -m pip install --upgrade pip setuptools wheel`
      2. `python -m pip install -r requirements`
   5. Download a vocabulary for spaCy.
      1. `python -m spacy download en_core_web_sm`
   6. I have a module for common spacy functions, install that.
      1. `python -m pip install git+https://github.com/rafelafrance/traiter.git@master#egg=traiter`
   7. Install the appropriate version of pytorch & pytorch vision. If your computer has an NVIDIA GPU I recommend the 1st line. If you do not have one, use the 2nd line.
      1. `python -m pip3 install -U torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
      2. `pip3 install torch torchvision`
   8. Create some useful directories.
      1. `mkdir -p data/models/flowering`
      2. `mkdir -p data/models/fruiting`
      3. `mkdir -p data/models/leaf_out`
      4. `mkdir -p data/output`
2. You need to do this every time you start using this module.
   1. `cd /path/to/herbarim_phenology`
   2. `source ./.venv/bin/activate`
