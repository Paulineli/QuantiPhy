input_dir="model_outputs"
output_dir="mra_results"
gt_file="quantiphy_validation.csv"

python evaluator.py $input_dir $output_dir --gt_file $gt_file
