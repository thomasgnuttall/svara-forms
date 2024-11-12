# Contextual Influences on the Performance of Svara

This repository contains the accompanying code and results for the paper

```Thomas Nuttall, Xavier Serra, Lara Pearson. "Svara-Forms in Carnatic Music: Contextual Influences on the Performance of Svara".```

Presented to the Workshop on Indian Music Analysis and Generative Applications (WIMAGA), a Satellite Workshop of the 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025).

## Usage

This analysis was developed using Python 3.9. To install all dependencies run.

`pip install -r requirements`

## Data
This analysis relies on two performances in raga Bhairavi from the Carnatic corpus of the Saraga dataset - Kamakshi (composed by Syama Sastri), performed by Sanjay Subrahmanyan, and Raksha Bettare (composed by Tyagaraja), performed by Shruthi S. Bhat. These performances are not provided in this repository and must be accessed via the [Saraga dataset](https://mtg.github.io/saraga/). 

The two audios corresponding to these performances should be stored at `data/audio/kamakshi.wav` and `data/audio/raksha_bettare.wav`. If you wish to use another location, you can alter the paths in [data/metadata.json](data/metadata.json). 

## Reproducing Presented Analysis

The scripts to reproduce this analysis can be found in [scripts/pipeline](scripts/pipeline). These scripts should be ran from the home directory e.g...

```
python scripts/pipeline/1_extract_pitch_track.py
```

A further README is found in that directory with more detailed instructions.

## Clustering Results

The final svara-form clusters are available in [results/final/clusters_full/](results/final/clusters_full/). That folder contains more information on the groupings and notes describing them from a Carnatic musicological perspective.

