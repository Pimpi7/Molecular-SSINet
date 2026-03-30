# Interpretable Molecule Generation: Applying Self-Supervised Discovering of Interpretable Features to Molecule Generation 

This work is ispired by those two papers: 
- MoleGuLAR: Molecule Generation Using Reinforcement Learning with 
Alternating Rewards. Manan Goel, Shampa Raghunathan, Siddhartha 
Laghuvarapu and U. Deva Priyakumar* 
- Shi, Wenjie, et al. "Self-supervised discovering of interpretable features for 
reinforcement learning." IEEE Transactions on Pattern Analysis and Machine 
Intelligence (2020). 

The idea is to address the classical black-box problem of the reinforcement learning 
processes in the field of the molecule generation, adding to the MoleGuLAR pipeline a 
block that learns how the molecule is been created. This could build an instrument 
that highlights the most important parts in the molecule structure, that most likely will 
be the ones the model pays more attention to. To create this block I think that could be 
adapted the mechanics used in the Shi paper, from the image field to the SMILES 
(essentially from 2D to 1D).  
For a starting point the idea is to implement the models provided in the repo of the first 
paper (https://github.com/devalab/MoleGuLAR/tree/master) and build an interpreter 
on them adapting the architecture proposed by Shi et al. If this reveals to be unfeasible 
the Goel et al pipeline could be implemented from scratch before applying the 
understandability architecture. 
