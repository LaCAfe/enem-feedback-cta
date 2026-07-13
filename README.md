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
```


## Dataset
This study uses the AES-ENEM dataset (Silveira et al., 2024):
https://huggingface.co/datasets/kamel-usp/aes_enem_dataset
Download the dataset and place `train-sourceAOnly.parquet` in a `parquet/` directory
before running the pipeline.

## Requirements
```
pip install -r requirements.txt
```
Models must be installed via Ollama (see `models.txt`):
```
ollama pull cnmoro/mistral_7b_portuguese:q2_K
ollama pull llama3.2:latest
```


## References
Anchiêta, R. T., Luz, A. I., Lopes, S. L., and Moura, R. S. (2025). A zero-shot prompting approach for automated feedback generation on ENEM essays. In Brazilian Symposium
on Multimedia and the Web (WebMedia), pages 511–515. SBC.

Bird, S., Klein, E., and Loper, E. (2009). Natural language processing with Python: analyzing text with the natural language toolkit. O’Reilly Media, Inc.

Silveira, I. C., Barbosa, A., and Mau´a, D. D. (2024). A new benchmark for automatic essay scoring in Portuguese. In Proceedings of the 16th International Conference on Computational Processing of Portuguese-Vol. 1, pages 228–237.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. (2019). Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.
