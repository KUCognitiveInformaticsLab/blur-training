python compute_plot_dist.py --server gpu2 --compute --plot -a alexnet --stimuli s-b --models vss
python compute_plot_dist.py --server gpu2 --compute --plot -a alexnet --stimuli h-l --models vss
python compute_plot_dist.py --server gpu2 --compute --plot -a alexnet --stimuli h-l --models test

python compute_plot_dist.py --server gpu2 --compute --plot -a alexnet --models mix_no-blur
python compute_plot_dist.py --server gpu2 --compute --plot -a alexnet --models mix_no-sharp

# plot sample
#python compute_plot_dist.py --machine local --plot -a alexnet --models mix_no-blur --full