# DragonSR

This project searches for symbolic equations from data using DragonSR.

## Install

From the `leaderboard` directory:

```bash
pip install -r requirements.txt
```

## Configuration

Create or edit `config.txt` with your dataset path and runtime settings.
The file usually contains flags such as:
- `data_path`
- `config_file_path`
- `loss_mode`
- `var_aug`
- `add_noise`
- `Dragon.SPAR_OP_GROUPS`

## Run

From the `leaderboard` directory:

```bash
python3 main.py --run_dragonsr --data_path /path/to/your.csv --config_file_path /path/to/config.txt
```