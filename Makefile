in_files = data/raw/testset1.txt data/raw/testset2.txt data/raw/testset2_repeat.txt data/raw/testset3.txt
out_files = data/interim/testset1.csv data/interim/testset2.csv data/interim/testset2_repeat.csv data/interim/testset3.csv

$(out_files): $(in_files)
	python scripts/create_data_with_labels.py

# Run just the four data files
data: $(out_files)
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --test False
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-3.5-chat --prompt chain --ordering 0 --test False
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-3.5-chat --prompt chain --ordering 1 --test False
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-3.5-chat --prompt chain --ordering 2 --test False

repeats: $(out_files)
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 1 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 2 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 3 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 4 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 5 --test True

repeats2: $(out_files)
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 6 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 7 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 8 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 9 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 10 --test True
	python scripts/get_odd_one_out.py --dataset data/interim/testset2.csv --model gpt-4 --prompt chain --ordering 11 --test True