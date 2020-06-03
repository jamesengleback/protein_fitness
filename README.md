# protein_fitness
This repositry explores the protein fitness landscape [**dataset**](https://github.com/jamesengleback/protein_fitness/blob/master/elife-16965-supp1-v4.xlsx) from [**Adaptation in protein fitness landscapes is facilitated by indirect paths**](https://github.com/jamesengleback/protein_fitness/blob/master/Adaptation%20in%20protein%20fitness%20landscapes%20is%20facilitated%20by%20indirect%20paths.pdf) (Wu *et al.* 2016). The dataset was also key in [**Machine learning-assisted directed protein evolution**](https://github.com/jamesengleback/protein_fitness/blob/master/Machine%20learning-assisted%20directed%20protein%20evolution.pdf) (Wu *et al.* 2020), where is was used to develop a machine learning-based technique to guide directed evolution experiments.

TODO: sort out refs

## Mutants
In the dataset, Wu *et al.* (2016) attempt to experimentally characterise a large mutant library of *Streptototal* G protein B1 (GB1) - an immunoglobulin-binding protein. For epistatic (interacting) mutation sites (V39, D40, G41 and V54) were chosen, and all possible combinations of mutants were generated using sauration mutagenesis (20^4, 160,000) as a pool. The procedure was engineered so that the mutant proteins would be fused to their parent mRNA, which would allow "fit" mutants to be identified after separation from "unfit" mutants.

TODO: image + more info of GB1

## Fitness - affinity selection by mRNA display
The **'fitness'** of GB1 was its binding affinity for **IgG-Fc**. Fitness was measured as the abundance of a mutant's fucion-mRNA sequences bound to IgG-Fc relative to its abundance before being mixed with the antibody.

TODO: image of selection process

## The dataset
[**elife-16965-supp1-v4.xlsx**](https://github.com/jamesengleback/protein_fitness/blob/master/elife-16965-supp1-v4.xlsx) is an excel sheet that contains 149361 entries, each with the following features:
* **Variants:** the mutant; in the format ```NNNN``` corresponding to the identity of the amino acids at V39, D40, G41 and V54 respectively
* **HD:** ??
* **Count input:** The number of copies of each mutant detected in the mutant pool before mixing with antibodies
* **Count selected:** The number of copies of each mutant detected in the antibody-bound fraction
* **Fitness:** ```Count selected```/```Count input```

TODO: check mutation correspondance
TODO: HD
TODO: elife-16965-supp2-v4.xlsx

## Notebooks
#### [**protein-fitness-perceptron-featureless.ipynb**](https://github.com/jamesengleback/protein_fitness/blob/master/protein-fitness-perceptron-featureless.ipynb)
* Confirm fitness is ```Count selected```/```Count input```
* Look at distribution of fitness scores
* Preprocess data - No feature engiineering, just one-hot encoding
* Build simple neural network and data-loader using ```pytorch```
* Train on half the dataset, evaluate on remaining half using Pearson correlation coefficient. TODO: |not normal distribution of fitness scores, find appropriate score
* Create a simple generative neural network to generate new one-hot encoded vectors, and a reverse encoding function.
* Train the generative neural network using the initial, fitness-evaluating neural network
* Generate new, fit sequences
* Fitness landscape visualisation attempts - use umap and autoencoder to create embeddings, plot in 2D heat map where 'heat' is fitness - looks very rugged/ugly
