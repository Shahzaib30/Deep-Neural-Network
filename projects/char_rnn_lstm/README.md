# Character-level LSTM (toy example)

This project implements a minimal character-level LSTM for experimentation.

Highlights

- Fetches a small Shakespeare dataset for toy experiments.
- Implements batching, a compact LSTM model, and a sampling helper.
- Intended for educational and experimental use; not production-ready.

Quick start

```bash
python projects/char_rnn_lstm.py
```

Notes

- Requires `torch` and `requests`.
- Consider adding argument parsing and checkpoints if training for longer.
