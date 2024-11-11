# Analysis Scripts

In this folder is a list of python scripts to reproduce the analysis in 

```
Thomas Nuttall, Xavier Serra, Lara Pearson. "Svara-Forms in Carnatic Music: Contextual Influences on the Performance of Svara".
```

The analysis relies on two performances in raga Bhairavi from the Carnatic corpus of the Saraga dataset - Kamakshi (composed by Syama Sastri), performed by Sanjay Subrahmanyan, and Raksha Bettare (composed by Tyagaraja), performed by Shruthi S. Bhat. These performances are not provided in this repository and must be accessed via the [Saraga dataset](https://mtg.github.io/saraga/). 

The two audios corresponding to these performances should be stored at `data/audio/kamakshi.wav` and `data/audio/raksha_bettare.wav`. If you wish to use another location, you can alter the paths in [data/metadata.json](data/metadata.json). 

The scripts should be ran in order from the [home directory](https://github.com/thomasgnuttall/svara-forms). Each script writes data objects to the paths defined in [data/metadata.json](data/metadata.json).

## Scripts

| **#** | **Script Name**     | **Function**                                                         |
|-------|---------------------|----------------------------------------------------------------------|
| 1     | Extract Pitch Track | Extract predominant pitch tracks for both performances               |
| 2     | Process Data        | Process pitch tracks - interpolate, smooth, and normalize            |
| 3     | Create Dataset      | Isolate individual svara observations - discard, keep, and transpose |
| 4     | DTW Distances       | Create DTW self similarity matrix for each svara group               |
| 5     | Gridsearch          | Gridsearch of clustering hyperparameters                             |
| 6     | Clustering          | Create final clusters using best hyperparameters                     |
| 7     | Context Analysis    | Analysis of relationship between cluster and melodic context         |
| 8     | Duration Analysis   | Analysis of relationship between cluster and duration                |

