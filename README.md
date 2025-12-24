<div align="center" style="font-family: charter;">

<h1><img src="assets/QuantiPhy_logo_pure.ico" width="3%"/> <i>QuantiPhy</i>:</br>A Quantitative Benchmark Evaluating Physical Reasoning Abilities</br> of Vision-Language Models</h1>

<br />

<a href="https://arxiv.org/pdf/2512.19526" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-QuantiPhy-red?logo=arxiv" height="20" />
</a>
<a href="https://quantiphy.stanford.edu/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-QuantiPhy-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation" target="_blank">
    <img alt="HF Dataset: QuantiPhy-validation" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-QuantiPhy_validation-ffc107?color=ffc107&logoColor=white" height="20" />
</a>


<div>
    <a href="https://www.linkedin.com/in/puyin-li-32709a299" target="_blank">Puyin Li</a><sup>1*</sup>,</span>
    <a href="https://ai.stanford.edu/~xtiange/" target="_blank">Tiange Xiang</a><sup>1*</sup>, </span>
    <a href="https://www.gsb.stanford.edu/programs/phd/academic-experience/students/ella-mao" target="_blank">Ella Mao</a><sup>1*</sup>,</span>
    <a href="https://www.linkedin.com/in/shirley-jinshan-wei-611545328" target="_blank">Shirley Wei</a><sup>1</sup>,</span>
    <a href="https://www.paprikachen.com" target="_blank">Xinye Chen</a><sup>1</sup>,</span>
    <a href="https://www.ust.com/en/boundless/boundless-thinkers/adnan-masood-phd" target="_blank">Adnan Masood</a><sup>2</sup>,</span>
    <a href="https://profiles.stanford.edu/fei-fei-li" target="_blank">Li Fei-Fei</a><sup>1†</sup>,</span>
    <a href="https://stanford.edu/~eadeli/" target="_blank">Ehsan Adeli</a><sup>1†</sup>,</span>
</div>

<div>
    <sup>1</sup>Stanford University&emsp;
    <sup>2</sup>UST&emsp;
</div>

<div>
    <sup>*</sup>Equal first authorship&emsp;
    <sup>†</sup>Equal last authorship&emsp;
</div>

<img src="assets/teaser.webp" width="100%"/>
<p align="justify"><i>On a crowded city street, a bird's nest falls from a branch, a car rushes by, an eagle flits over a building, and a person walks in a crosswalk — the real world is full of complex physical motion. To enable AI to understand and navigate this environment, it is essential for generalist embodied systems to reason about physical properties quantitatively. Because objects obey common laws of physics, their kinematic properties (such as size, velocity, and acceleration) are interrelated. This interdependence makes it possible for visual AI to systematically reason about these properties with respect to available priors. In this work, we present \textsc{QuantiPhy}, the first benchmark to evaluate the reasoning ability of AI models on quantitative kinematic inference tasks.</i></p>

</div>

## Overview

This repository contains the evaluation code for the **QuantiPhy** benchmark. It calculates the Multi-Region Accuracy (MRA) metric for quantitative physical reasoning tasks. The evaluation script processes model output CSV files, compares the predicted values against ground truth, and computes MRA scores across different difficulty levels (theta thresholds) and categories (S2, D2, S3, D3).


## Directory Structure

Ensure your directory is organized as follows:

```
QuantiPhy/
├── evaluator.py              # Main evaluation script
├── evaluate.sh               # The script to launch evaluation
├── model_outputs/            # Directory for model prediction CSVs
│   ├── model_A.csv
│   └── model_B.csv
│   └── ...
└── mra_results/              # Directory where results will be saved
```

## Use case 1: Evaluation on QuantiPhy-validation

1. **Obtain Data**: Follow the instructions on our [HuggingFace page](https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation) to download the necessary data needed for evaluating on QuantiPhy-validation:
   * `validation_videos`: The folder contains videos in the validation set.
   * `quantiphy_validation.csv`: Metadata, question, prior, and ground truth annotations in the validation set. Put the CSV file anywhere you like, but **remember to update the `gt_file` variable in `evaluate.sh`.**


2. **Obtain Results**: For all items in the validation CSV file, run your VLM with the corresponding `question` and `prior` along with the linked video to obtain the results. The results should follow the same structure as the example output in `model_outputs/gpt-5.1.csv`.



3. **Prepare Outputs**: Place your model prediction CSV files in the `model_outputs/` directory. Each CSV file should contain at least the following columns:
   - `video_id`
   - `question`
   - `parsed_value` (The numeric value extracted from the model's response)
   - `video_type` (Optional, used for categorization)
   - `inference_type` (Optional, used for categorization)

4. **Run Evaluation**: Execute the provided shell script to start the evaluation:

   > **Note**: Please update `evaluate.sh` with your local paths before running evaluation.

   ```bash
   bash evaluate.sh
   ```

   This script will:
   - Read all CSV files from `model_outputs/`.
   - Use `quantiphy_validation.csv` as the ground truth.
   - Save the aggregated results to `mra_results/all_model_results.csv`.

## Output

The evaluation results are saved in `mra_results/all_model_results.csv`. This CSV file contains:

- **model**: Name of the model (derived from filename).
- **mra_average**: Average MRA score across all categories.
- **mra_S2, mra_D2, mra_S3, mra_D3**: Specific MRA scores for each category.
- **mra_bg_***: MRA scores broken down by background type.
- **mra_obj_***: MRA scores broken down by object number (single vs multiple).
- **invalid_percentage**: Percentage of responses where a valid numeric value could not be parsed.


## Citation

If you find this work useful in your research or project, please cite:

```bibtex
@article{li2025quantiphy,
      title   = {QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models},
      author  = {Li, Puyin and Xiang, Tiange and Mao, Ella and Wei, Shirley and Chen, Xinye and Masood, Adnan and Li, Fei-Fei and Adeli, Ehsan},
      journal = {arXiv preprint arXiv:2512.19526},
      year    = {2025}
    }
```