# alexnet
python compute_plot_tSNE.py --compute --plot --stimuli s-b -a alexnet --num_classes 16 --models mix_no-blur
python compute_plot_tSNE.py --compute --plot --stimuli s-b -a alexnet --num_classes 16 --models mix_no-sharp

# vone_alexnet
python compute_plot_tSNE.py --compute --plot --stimuli s-b -a vone_alexnet --num_classes 16 --models mix_no-blur
python compute_plot_tSNE.py --compute --plot --stimuli s-b -a vone_alexnet --num_classes 16 --models mix_no-sharp

# hyper-parameters
## dim == 3 (plot them later)
python compute_plot_tSNE.py --compute --num_dim 3 --stimuli s-b -a alexnet --num_classes 16 --models test
python compute_plot_tSNE.py --compute --num_dim 3 --stimuli s-b -a alexnet --num_classes 1000 --models test
python compute_plot_tSNE.py --compute --num_dim 3 --stimuli s-b -a vone_alexnet --num_classes 16 --models test
python compute_plot_tSNE.py --compute --num_dim 3 --stimuli s-b -a vone_alexnet --num_classes 1000 --models test

## other parameters
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 10 --n_iter 500
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 10 --n_iter 1000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 10 --n_iter 5000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 10 --n_iter 10000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 10 --n_iter 20000

python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 30 --n_iter 500
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 30 --n_iter 1000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 30 --n_iter 5000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 30 --n_iter 10000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 30 --n_iter 20000

python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 50 --n_iter 500
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 50 --n_iter 1000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 50 --n_iter 5000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 50 --n_iter 10000
python compute_plot_tSNE.py --models test --compute --plot --stimuli s-b --perplexity 50 --n_iter 20000
