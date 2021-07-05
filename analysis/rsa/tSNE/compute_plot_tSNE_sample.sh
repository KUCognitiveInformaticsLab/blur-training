# alexnet
#python compute_plot_tSNE.py --compute --plot --stimuli s-b -a alexnet --num_classes 16 --models vss
#python compute_plot_tSNE.py --compute --plot --stimuli s-b -a alexnet --num_classes 1000 --models vss
# vone_alexnet
#python compute_plot_tSNE.py --compute --plot --stimuli s-b -a vone_alexnet --num_classes 16 --models vss
#python compute_plot_tSNE.py --compute --plot --stimuli s-b -a vone_alexnet --num_classes 1000 --models vss

# alexnet
python compute_plot_tSNE.py --compute --plot --stimuli s-b -a alexnet --num_classes 16 --models mix_no-blur
python compute_plot_tSNE.py --compute --plot --stimuli s-b -a alexnet --num_classes 16 --models mix_no-sharp

# vone_alexnet
python compute_plot_tSNE.py --compute --plot --stimuli s-b -a vone_alexnet --num_classes 16 --models mix_no-blur
python compute_plot_tSNE.py --compute --plot --stimuli s-b -a vone_alexnet --num_classes 16 --models mix_no-sharp

# plot example
#python compute_plot_tSNE.py --plot --stimuli s-b -a alexnet --num_classes 16 --models vss --machine local
#python compute_plot_tSNE.py --plot --stimuli s-b -a vone_alexnet --num_classes 1000 --models vss --machine local
