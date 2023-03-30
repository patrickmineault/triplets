in_files = data/raw/testset1.txt data/raw/testset2.txt data/raw/testset2_repeat.txt data/raw/testset3.txt
out_files = data/interim/testset1.csv data/interim/testset2.csv data/interim/testset2_repeat.csv data/interim/testset3.csv

$(out_files): $(in_files)
	python scripts/create_data_with_labels.py

# Run just the four data files
data: $(out_files)
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-3.5-chat --prompt chain --order 0
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-3.5-chat --prompt chain --order 1
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-3.5-chat --prompt chain --order 2