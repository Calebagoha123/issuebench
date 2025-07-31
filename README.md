# IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance

**Contact**: [REDACTED]


## Repo Structure

```
├── 1_dataset_construction
│   ├── 1_preprocessing                 # downloading and cleaning source data
│   ├── 2_relevance_filtering           # filtering prompts for relevance
│   ├── 3_writing_assistance_filtering  # filtering prompts for writing assistance
│   ├── 4_extracting_issues             # clustering prompts to extract issues
│   └── 5_extracting_templates          # extracting templates from prompts
│
├── 2_final_dataset
│   └── prompt_ingredients              # issues and templates for IssueBench
│
└── 3_experiments
    ├── 1_stance_classifier_evaluation  # evaluating stance classifiers
    ├── 2_inference                     # scripts to collect results on IssueBench
    └── 3_analysis                      # notebooks to reproduce analysis from our paper
```

**Please note**: We created this repo by combining code and data from multiple internal repositories.
Some paths in some scripts may need to be adjusted.
If you have any questions, please feel free to reach out to us.
We are happy to help!


## Using IssueBench

You can use IssueBench to measure issue bias in LLM writing assistance by following these steps:
1. Download the full IssueBench dataset from Hugging Face.
2. Generate completions on IssueBench using your LLM of choice.
3. Classify the stance of these completions based on the taxonomy described in our paper.
4. Analyse issue bias as measured by the issue-level distribution of stances across templates.

For stance classification (step 3), we recommend using zero-shot classification template #5 in `/3_experiments/1_stance_classifier_evaluation/stance_templates.csv` paired with the best LLM you can afford to run.

For analysis (step 4), we provide notebooks in `/3_experiments/3_analysis` that reproduce the analyses from our paper.
The model responses we collected for our paper (~3m per model) are available on Hugging Face.

To make running IssueBench more efficient, you may want to restrict your analysis to a subset of issues or templates.
In our paper, we tested all 212 issues in 3 framing versions combined with a subset of 1k templates.


## Adapting IssueBench

You can easily adapt IssueBench to include new issues or templates. 
Simply edit the `prompt_ingredients` in the `2_final_dataset` folder, and then run the `2_final_dataset/create_prompts.ipynb` script to generate new prompts.

## License Information

The IssueBench dataset is licensed under CC-BY-4.0 license.
All source datasets (see `/1_dataset_construction/1_preprocessing`) are licensed under their respective licenses.
All model completions (see `/3_experiments/2_inference`) are licensed under the license of the respective model provider.

## Citation Information

If you use IssueBench, please cite our paper:
 
[REDACTED]