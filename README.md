# Investigating the Zone of Proximal Development of Language Models for In-Context Learning.
> **Abstract:** In this paper, we introduce a learning analytics framework to analyze the in-context learning (ICL) behavior of large language models (LLMs) through the lens of the Zone of Proximal Development (ZPD), an established theory in educational psychology. ZPD delineates the space between what a learner is capable of doing unsupported and what the learner cannot do even with support. We adapt this concept to ICL, measuring the ZPD of LLMs based on model performance on individual examples with and without ICL. Furthermore, we propose an item response theory (IRT) model to predict the distribution of zones for LLMs. Our findings reveal a series of intricate and multifaceted behaviors of ICL, providing new insights into understanding and leveraging this technique. Finally, we demonstrate how our framework can enhance LLM in both inference and fine-tuning scenarios: (1) By predicting a model's zone of proximal development, we selectively apply ICL to queries that are most likely to benefit from demonstrations, achieving a better balance between inference cost and performance; (2) We propose a human-like curriculum for fine-tuning, which prioritizes examples within the model's ZPD. The curriculum results in improved performance, and we explain its effectiveness through an analysis of the training dynamics of LLMs.

This repo contains the source code and a guideline for the above [paper](https://arxiv.org/abs/2502.06990) accepted at NAACL 2025 findings. 

## Meausure the ZPD of LLMs

1. Generate Oracle in-context demontsrations

```bash 
python -m zpd.gen_ices --dataset=<"gsm8k or ez_stance"> --ice_type="oracle" --num_candidates=<int> --num_ices=<int>
```

2. Run inference with and without ICL

```bash
python -m zpd.inference --dataset="dataset" 			# gsm8k or ez_stance
						--model_name="model_name" 		# e.g. meta-llama/Llama-2-7b-hf
						--icl_strategy="icl_strategy" 	# oralce, similarity, random, etc
						--model_ckpt="model_ckpt"  		# if run inference with a finetuned model
						--max_new_tokens=250 	   		# number of max generated tokens
						--split="train, test, dev"
```

3. Divide zones according to model's performance from Step 2. 
```bash
python -m zpd.zones --dataset=<"gsm8k/ez_stance">
```

--- 

## Zone Prediction with IRT

1. Prepare the data for IRT

```bash
python -m pred.irt_data --dataset=<"dataset"> --job="gen_splits"		# convert data to IRT format and create splits
python -m pred.irt_data --dataset=<"dataset"> --job="gen_embeddings" 	# embeddings for MultiIRT
```

2. Train IRT models
```bash
	python -m pred.irt_model --dataset="dataset" 			# gsm8k or ezstance 
						 --use_answer="True"  
						 --irt_model_type=<"irt_model">  	# [1pl, 2pl, mirt]  
						 --enable_gamma='True'   			# True=icl_irt
						 --eval_metrics='roc_auc'  			# [roc_auc, f1_score, accuracy]  
						 --trait_dim=32  					# only for multi-dimension IRT  
						 --job="train"  					# [train, eval]
```

---

## Use Case
We demonstrate two applications of our framework in the paper. 

1. Selective ICL

```bash
python -m app.adaptiveICL.adaptive_icl
```
Arguments are specified in the `get_adaptive_icl()` function. 

2.  ZPD-based Curriculum

```bash
python -m app.curriculum.train 	--model_name 	# model to train.
								--dataset		# gsm8k or ez_stance
								--curriculum  	# [zpd, random]
```
Run `zpd/inference.py` (Step 2) and `zpd/evaluate.py` to evaluate its performance. 


If you find our work or code helpful, please cite our paper as:
```
@article{cui2025investigating,
  title={Investigating the Zone of Proximal Development of Language Models for In-Context Learning},
  author={Cui, Peng and Sachan, Mrinmaya},
  journal={arXiv preprint arXiv:2502.06990},
  year={2025}
}
```

If you have any questions, do not hesitate to contact peng.cui@inf.ethz.ch!