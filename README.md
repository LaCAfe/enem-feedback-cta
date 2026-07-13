## ENEM Essay Feedback Using Evaluator Term Set (CTA)

This repository contains the pipeline for automated feedback generation
for ENEM Competency 5 (intervention proposal). The method augments
zero-shot prompts with an Evaluator Term Set (CTA) extracted from human
evaluator comments, and evaluates the improvement using BERTScore F1
and the Wilcoxon signed-rank test.


When using this repository for academic purposes, please cite:

Carvalho, F., Soares, V., Bezerra, E., and Guedes, G. (2026).
ENEM Essay Feedback Using LLM-Augmented Prompts.
In *Anais do STIL 2026*. SBC.

```bibtex
@inproceedings{carvalho2026enem,
  title     = {{ENEM} Essay Feedback Using {LLM}-Augmented Prompts},
  author    = {Carvalho, Flavio and Soares, Vanessa and Bezerra, Eduardo and Guedes, Gustavo},
  booktitle = {Anais do 17º Simpósio em Tecnologia da Informação e da Linguagem Humana (STIL)},
  year      = {2026},
  publisher = {SBC}
}
