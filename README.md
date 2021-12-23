# Herbarium phenology![Python application](https://github.com/rafelafrance/herbarium_phenology/workflows/CI/badge.svg)

## Extracting phenological information from digitized herbarium specimens

TODO list:
- Scrape angiosperm class and family data from authoritative websites.
- Load iDigBio data. (I am currently using data from another project.)
- ~~Filter iDigBio data to only include angiosperms~~
- ~~Get flowering, fruiting, and leaf-out information from iDigBio fields using a variety of NLP techniques.~~
- ~~Train a neural network(s) to classify images as flowering, fruiting, and leaf-out.~~
- Download image data targeted at underrepresented traits (flowering, fruiting, etc.) and phylogenetic orders.
- Use semi-supervised learning to build up data for missing traits. Most herbarium records have data about one trait, we want to train a model to recognize all traits at once.
- Prepare data for showing the results, false (& true) positives and negatives.
- Fun & games with the model architecture.
- Create detailed notes of what we're doing and make it the new README.

This project extends Brian Stucky's work located [here](https://gitlab.com/stuckyb/herbarium_phenology).
