# MLOps Task


The goal wasn’t to build a complex ML model, but to show how a simple data pipeline can be written in a clean and production-style way — with proper config handling, logging, metrics, and Docker support.

# What the script does

The program reads a CSV file that contains market data (OHLCV).

From that data:
- It uses the `close` price
- Calculates a rolling average (based on a window from config)
- Compares current price with the rolling average
- Generates a simple signal:
  - 1 → price is above average  
  - 0 → price is below average  

At the end, it outputs some basic metrics like:
- how many rows were processed
- signal rate
- how long the job took

## Why I built it this way

I tried to keep things simple but realistic.

In real systems:
- configs should not be hardcoded
- runs should be reproducible
- logs should help debug issues
- output should be structured (not random prints)

## How to run

### Run locally

```bash
pip install -r requirements.txt

python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
