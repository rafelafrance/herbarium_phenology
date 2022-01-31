# Herbarium phenology![Python application](https://github.com/rafelafrance/herbarium_phenology/workflows/CI/badge.svg)

## Extracting phenological information from digitized herbarium specimens

TODO list:
- Scrape angiosperm class and family data from authoritative websites.
- Load iDigBio data. (I am currently using data from another project.)
- ~~Filter iDigBio data to only include angiosperms~~
- ~~Get flowering, fruiting, and leaf-out information from iDigBio fields using a variety of NLP techniques.~~
- ~~Train a neural network(s) to classify images as flowering, fruiting, and leaf-out.~~
- ~~Download image data targeted at underrepresented traits (flowering, fruiting, etc.) and phylogenetic orders.~~
- ~~Use semi-supervised learning to build up data for missing traits. Most herbarium records have annotations about one trait.~~
- Fun & games with the model architecture.
- Create detailed notes of what we're doing and link to them in the README.

This project extends Brian Stucky's work located [here](https://gitlab.com/stuckyb/herbarium_phenology).

## Setup

**TODO**: This is a tad bit complicated.

1. Create a virtual environment
   1. Make sure you have a virtual environment manager installed. I use `virtualenv`.
      1. `pip install --user virtualenv`
   2. Check out a tag.
      1. `cd /path/to/herbarim_phenology`
      2. `git checkout v0.1` (or another tag)
   2. Create a virtual environment.
      2. `virtualenv -p python3.9 .venv` (You may use python 3.9+)
      3. `source ./.venv/bin/activate`
   3. Install module requirements.
      1. `python -m pip install --upgrade pip setuptools wheel`
      2. `python -m pip install -r requirements`
   4. Download a vocabulary for spaCy.
      1. `python -m spacy download en_core_web_sm`
   5. I have a module for common spacy functions, install that.
      1. `python -m pip install git+https://github.com/rafelafrance/traiter.git@master#egg=traiter`
   6. Install the appropriate version of pytorch & pytorch vision. If your computer has an NVIDIA GPU I recommend the 1st line. If you do not have one, use the 2nd line.
      1. `python -m pip3 install -U torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
      2. `pip3 install torch torchvision`
   7. Create some useful directories.
      1. `mkdir -p data/models/flowering`
      1. `mkdir -p data/models/fruiting`
      1. `mkdir -p data/models/leaf_out`
      2. `mkdir -p data/output`
2. You need to do this every time you start using this module.
   1. `cd /path/to/herbarim_phenology`
   2. `source ./.venv/bin/activate`
