# Q-Learning_Tabular_and_Deep
[//]: # (TODO: Maybe add more explanation about the assignment?)

First, generate 5 sets of results:
```sh
python train.py --steps 1000000 --seed 0 --run_name naive
python train.py --steps 1000000 --seed 1 --run_name naive
python train.py --steps 1000000 --seed 2 --run_name naive
python train.py --steps 1000000 --seed 3 --run_name naive
python train.py --steps 1000000 --seed 4 --run_name naive
```

[//]: # (TODO: won't always be 'naive', probably need to do something inside of ablation)
Then, extract all npz files inside of results/naive, make sure they are extracted in directories of the name format `seed_{n}_.npz_FILES` where `n` is the seed number given by the previous commands.

After this, run `python plot.py` to generate a graph of the results over the steps.