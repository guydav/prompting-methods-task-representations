
# Do different prompting methods yield a common task representation in language models?

Code and some of the data for our paper "Do different prompting methods yield a common task representation in language models?": https://arxiv.org/abs/2505.12075

## Table of Contents
1. [Setup](#setup)
2. [Preliminary steps](#preliminary-steps)
3. [Experiments](#experiments)
4. [Data analysis](#data-analysis)
5. [Thanks](#thanks)
6. [Citation](#citation)

## Setup

The file `fv_env.yaml` provides the output of `conda env export` on the conda environment we used. We also provide `requirements.txt` files. Support for Windows/OSX is limited due to to CUDA dependencies in `bitsandbytes`. 

Please set up the environment variable `STORAGE_ROOT` to path to the folder where you would like results to be saved. It should include:
- `dataset_files`: the original dataset files, downloaded from https://github.com/ericwtodd/function_vectors/tree/main/dataset_files
- `full_results_top_heads`: a folder to save the top heads identified by the function vector extraction process. Our copy is provided in a folder with the same name in the project root.
- `prompts`: the model-generated instructions for each task. If you do not wish to generate them anew, extract the `prompts.tar.gz` archive provided here.
- `short_real_text_caches`: a folder to store our cached real texts for the **real texts** baseline. You can either generate these yourself using `recipe/function_vectors/cache_short_texts.py`, or extract the `short_real_text_caches.tar.gz` archive we provide.

Once you have the environment and folder structure set up, you can begin to run experiments. 

### Hydra
Most executable scripts in our codebase also have a [Hydra](https://hydra.cc/docs/intro/) wrapper for submitting jobs. We provide configurations under the `configs` folder and one for local execution as `configs/mode/local.yaml`. If you have access to a slurm cluster and wish to run experiments on it instead, adapt `configs/mode/cluster.yaml` to your needs. Otherwise, use the `local` mode. 

In some cases, in the Hydra commands, we set the values of certain variables to `ignore`. This is to indicate they are not used by the script invoked, but exist for compatability with our overall Hydra configuration.

### Environment variable specifications
For all run commands provided below, we assume the following environment variables are defined (or are find-replaced in the provided commands). One easy to accomplish this is to prepend to the run command `VAR1=foo; VAR2=bar; <command>`. Relevant variables:
- `MODEL`: The [huggingface](https://huggingface.co/) IDs of one or more models, comma-separted without spaces.
- `DATASET`: The names of one or more datasets as they appear in subdirectories of the `dataset_files` folder specified above.
- `MODE`: either `local` or `cluster`
    - If running with the `cluster` mode, you should also specify the timeout as in the commands below; if not, you should remove it.
- `PROMPT_TYPE`: the instruction length, either `short`, `long`, or both,`short,long`
- `BASELINE`: one or more of the following, comma-separated with no spaces: `real_text`, `equiprobable`, `other_task_prompt`

## Preliminary steps

The steps below are both optional, one in case you decide not to use our cached short real texts (see above under Setup), and one in case you want to run multiple experiments with the asme model on the same dataset in parallel.

### Short real text caching

Cache short strings from a natural language dataset (in our case, `WikiText-103-v1`) and record their log-probabilities under a the model provided.

```bash
python cache_texts_main.py -m mode=${MODE} timeout=720 "prompt_type=ignore" "prompt_baseline=ignore" "dataset=ignore" "model=${MODEL}"+max_length_tokens=16,64"  
```

### Instruction generation
If not using our generated instructions provided in `prompts.tar.gz`, you have to generate these yourself. To do so, run the following command:

```bash
python -m recipe.function_vectors.generate_prompts_for_dataset  --prompt_template_key ${KEY} --dataset_name "${DATASET1}" "${DATASET2}" ... "${DATASETN}"
```

Where `KEY` is one of `short` or `long` and `DATASET1` through `DATASETN` are space-separated names of datasets.


### Best instruction filtering

When running multiple experiments in parallel with the same task/dataset and the same model, we found it's helpful to run the instruction filtering step in advance, as (1) it's required for all of them, and any job that's launched and doesn't find the expected output file will run it (which could lead to it being duplicated across jobs), and (2) it can run at a higher batch size than some other steps, so running it separately makes it easier to run with a higher batch size without specifying different batch sizes for different steps.

To do so, run the command below:

```bash
python prompt_filter_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "model=${MODEL}" "dataset=${DATASET}" 
```

## Experiments 

### Demonstration FV reproduction
To generate results with demonstration FVs (similar to Todd et al., 2024), run the following command:

```bash
python icl_fv_main.py -m mode=${MODE} timeout=720  "model=${MODEL}" "dataset=${DATASET}" 
```

After doing this, either use our set of top heads, or generate your own using the logic in `notebooks/compute_top_heads.ipynb`.Then, run the following with the 'universal' (overall top) heads:

```bash
python icl_fv_main.py -m mode=cluster timeout=720 "+universal_set=1" "model=${MODEL}" "dataset=${DATASET}"  &
```

### Instruction FVs without using our top heads
If you wish to generate your own set of tops heads, you first need to run the experiments without the universal (= global top heads) function vectors. To do so, run the command below:

```bash
python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=${BASELINE}" "model=${MODEL}" "dataset=${DATASET}" 
```

### Finding 1: overall instruction FV results
Run the evaluation with the universal set of top instruction FV heads. The value of the argument for `prompt_baseline` doesn't matter but it must be provided and be one of the valid options.

```bash
python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1"
```

### Finding 2: intervening with both function vectors
Run the evaluation set, intervening with both function vectors:

```bash
python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1" "+joint_intervention=1"
``` 

### Finding 3: shared attention heads
Does not require any additional experiments to run, given the mean activations and top heads generated above.


### Finding 4: `incongruent' function vectors
Run the same experiment script with the following flags:

```bash
python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1" "+use_icl_top_heads=1"

python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1" "+use_icl_mean_activations=1"

``` 

#### Finding 4 controls: least important heads and bottom heads
First, either use our sets of bottom and least important heads, or generate them using the logic in `notebooks/compute_top_heads.ipynb`.

Then, run the following experiments: 

```bash
# Least improtant ("min abs"olute causal effect) heads
python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1" "+use_min_abs_heads_prompt=1"

python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1" "+use_min_abs_heads_icl=1"

# Bottom heads
python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1" "+use_bottom_heads_prompt=1"

python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1" "+use_bottom_heads_icl=1"
```


### Finding 5: post-trained model instruction FV in base model
Run the following experiments, making sure to set `$MODEL` to only a set of base models:

```bash
python prompt_fv_main.py -m mode=${MODE} timeout=1440 "prompt_type=${PROMPT_TYPE}" "+cache_prompt_prefixes=1" "prompt_baseline=real_text" "model=${MODEL}" "dataset=${DATASET}" "+universal_set=1" "+use_instruct_model_fv=1"
```

The remainder of Finding 5 (Figure 4B) is generated using running the Finding 1 experiments with the OLMo-2-1124-7B suite of models.

## Data analysis

We perform our data analysis in the notebooks provided in the `notebook` directory:
- `compute_top_heads`: generate the sets of top heads
- `paper_examples`: generate example prompts and baselines for Appendix tables 2 and 3.
- `results_`:
    - `controls`: the appendix controls for Finding 4 (e.g., Appendix J.1)
    - `joint_localizers`: figures for Findings 1, 2, and 4
    - `overall`: figures for Finding 5 (and some auxiliary ones)
    - `top_heads`: figures for Finding 3


## Thanks

We thank Eric Todd for making code available for their previous paper, "Function Vectors in Large Language Models," which helped inspire our work. This repository started from their codebase adapted to a different template. If you find our work useful, please also see their [repository](https://github.com/ericwtodd/function_vectors) and [project website](https://functions.baulab.info/).


## Citation

To cite our work, please use the following entry:

```bibtex
@ARTICLE{Davidson2025,
  title         = "Do different prompting methods yield a common task
                   representation in language models?",
  author        = "Davidson, Guy and Gureckis, Todd M and Lake, Brenden M and
                   Williams, Adina",
  journal       = "arXiv [cs.CL]",
  abstract      = "Demonstrations and instructions are two primary approaches
                   for prompting language models to perform in-context learning
                   (ICL) tasks. Do identical tasks elicited in different ways
                   result in similar representations of the task? An improved
                   understanding of task representation mechanisms would offer
                   interpretability insights and may aid in steering models. We
                   study this through function vectors, recently proposed as a
                   mechanism to extract few-shot ICL task representations. We
                   generalize function vectors to alternative task
                   presentations, focusing on short textual instruction prompts,
                   and successfully extract instruction function vectors that
                   promote zero-shot task accuracy. We find evidence that
                   demonstration- and instruction-based function vectors
                   leverage different model components, and offer several
                   controls to dissociate their contributions to task
                   performance. Our results suggest that different task
                   presentations do not induce a common task representation but
                   elicit different, partly overlapping mechanisms. Our findings
                   offer principled support to the practice of combining textual
                   instructions and task demonstrations, imply challenges in
                   universally monitoring task inference across presentation
                   forms, and encourage further examinations of LLM task
                   inference mechanisms.",
  month         =  may,
  year          =  2025,
  archivePrefix = "arXiv",
  primaryClass  = "cs.CL"
}
```
