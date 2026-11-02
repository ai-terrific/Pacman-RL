# Pacman-Project

A Pacman environment with tabular Q-learning.

Teammates: Guanhui Guan, Jiahao Li, Jingyi Qian, Yuekun Guo, Qiao Zhang

References:

- http://ai.berkeley.edu/reinforcement.html
- https://github.com/gauravmittal1995/Pyman

## Requirements

- Python 3
- NumPy
- pygame (optional for headless training)
- Matplotlib (optional, for plots)

## Run

Run the CLI through `Pacman.py`:

```powershell
python Pacman.py --grid mediumGrid --train 5000 --test 50
```

Train without opening a game window:

```powershell
python Pacman.py --grid mediumGrid --train 5000 --test 0 --headless
```

Evaluate an existing checkpoint without training:

```powershell
python Pacman.py --grid mediumGrid --train 0 --test 50
```

Continue training from the checkpoint:

```powershell
python Pacman.py --grid mediumGrid --train 5000 --resume
```

During training, click **SAVE Q TABLE** at the bottom of the game window to save
a checkpoint to `pacman_checkpoint.pkl`. Checkpoints are not saved automatically
when training ends or the window closes. Use `--checkpoint` to select another
checkpoint path. Only load checkpoint files you created yourself.

## Options

- `--grid`: `miniGrid`, `xsmallGrid`, `smallGrid`, `mediumGrid`, or `largeGrid`
- `--ghosts`: number of ghosts
- `--train`, `--test`: training and evaluation episode counts
- `--epsilon`, `--gamma`, `--lr`: Q-learning parameters
- `--max-steps`: maximum steps per episode
- `--seed`: random seed for repeatable runs
- `--headless`: train/evaluate without pygame rendering
- `--render-every`: render every Nth training episode
- `--train-delay`, `--test-delay`: rendered frame delays in milliseconds;
  press `-` or `+` while a window is active to adjust the speed
- `--resume`: continue training from an existing checkpoint
- `--plot`: plot training win rate and score

Tabular Q-learning is most practical on `miniGrid`, `xsmallGrid`, and
`smallGrid`. The medium and large maps have many possible food configurations
and may require substantial training.

The Q-table state uses Pacman's position, the direction and maze distance to
the nearest food and ghost, and the number of currently safe next moves. This
compact representation does not distinguish every remaining-food layout, so
different situations can share a Q-table entry. Checkpoints made with the
previous, food-layout-based state representation are incompatible.
