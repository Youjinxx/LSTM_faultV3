# config.py

MODEL = dict(
    input_size=3,
    hidden_size=32,
    num_layers=2,
    output_size=2,
    dropout=0.3
)

TRAIN = dict(
    batch_size=32,
    num_epochs=100000,
    learning_rate=1e-2,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    loss_threshold=0.005,
    patience=3
)
