# Empathic Reactions. Modeling Empathy and Distress in Reaction to News Stories

This repository contains the dataset, experimental code and results presented in our EMNLP 2018.


## Dataset
Our dataset comprises 1860 short texts together with ratings for two kinds of empathic states, empathic concern and personal distress. It is, to our knowledge, the first publicly available gold standard for NLP-based empathy prediction. The `csv`-formatted data can be found [here](data/responses/data/messages.csv). For details regarding our annotation methodology please refer to the paper.

## License
Our dataset is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Re-Running the Experiments
We ran our code under Ubuntu 16.04.4. Our `conda` environment is specified in `environment.yaml`. To re-run our experiments, you have to add the root directory of the repository to you python path and setup an environment variable `VECTORS`. Details can be found in `activate_project_environment` and `constants.py`. Please note that re-running our code will produce varying results due to racing conditions caused by multi-threading.

The FastText word vectors can be found [here](https://fasttext.cc/docs/en/english-vectors.html).

Once everything is set up, executing `run_experiments.sh` will re-run our cross-validation experiment. The results will be stored in `modeling/main/crossvalidation/results`.

## Paper & Citation

```
@inproceedings{Buechel18emnlp,
author={Buechel, Sven and Buffone, Anneke and Slaff, Barry and Ungar, Lyle and Sedoc, Jo{\~{a}}o},
title = {Modeling Empathy and Distress in Reaction to News Stories},
year = {2018}
booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018)}
}
```

You can find our arXiv preprint [here](https://arxiv.org/pdf/1808.10399.pdf).

## Contact
I am happy to give additional information or get feedback about our work via email: sven.buechel@uni-jena.de