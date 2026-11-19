"""Запускает эксперимент на MNIST и сохраняет метрики в CSV и JSON."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mnist_numpy.data import DEFAULT_DATA_DIR, SPLIT_SEED, load_test, load_train_val
from mnist_numpy.loss import softmax_cross_entropy
from mnist_numpy.model import MLP
from mnist_numpy.training import train_epoch


def evaluate(model, x, labels):
    logits = model.forward(x)
    loss, _ = softmax_cross_entropy(logits, labels)
    predicted = np.argmax(logits, axis=1)
    return float(loss), float(np.mean(predicted == labels)), predicted


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def run(args):
    (x_train, y_train), (x_val, y_val) = load_train_val(args.data_dir, args.limit_train)
    model = MLP(x_train.shape[1], args.hidden_size, 10, np.random.default_rng(args.seed))
    shuffle_rng = np.random.default_rng(args.seed)
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, x_train, y_train, args.batch_size, args.learning_rate, shuffle_rng
        )
        val_loss, val_accuracy, _ = evaluate(model, x_val, y_val)
        history.append((epoch, float(train_loss), val_loss, val_accuracy))
        print(
            f"Эпоха {epoch:02d} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | val accuracy {val_accuracy:.2%}",
            flush=True,
        )

    metrics = {
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "split_seed": SPLIT_SEED,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    if args.test:
        x_test, y_test = load_test(args.data_dir)
        test_loss, test_accuracy, predicted = evaluate(model, x_test, y_test)
        metrics.update(test_loss=test_loss, test_accuracy=test_accuracy, test_size=len(y_test))
        print(f"Тест | loss {test_loss:.4f} | accuracy {test_accuracy:.2%}", flush=True)

    args.run_dir.mkdir(parents=True)
    write_csv(args.run_dir / "history.csv", ("epoch", "train_loss", "val_loss", "val_accuracy"), history)
    (args.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    if args.test:
        matrix = np.bincount(y_test * 10 + predicted, minlength=100).reshape(10, 10)
        write_csv(
            args.run_dir / "confusion.csv",
            ("true\\pred", *range(10)),
            ((label, *row) for label, row in enumerate(matrix)),
        )
        errors = np.flatnonzero(y_test != predicted)
        write_csv(
            args.run_dir / "mistakes.csv",
            ("test_index", "true_label", "predicted_label"),
            ((index, y_test[index], predicted[index]) for index in errors),
        )

    print(f"Результаты сохранены в {args.run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/base"), help="Новый каталог для результатов")
    parser.add_argument("--hidden-size", type=int, default=64, help="Число скрытых нейронов")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, help="Число обучающих объектов для быстрой проверки")
    parser.add_argument("--test", action="store_true", help="В конце посчитать метрики на тестовой выборке")
    args = parser.parse_args()

    if min(args.hidden_size, args.epochs, args.batch_size) <= 0:
        parser.error("Число скрытых нейронов, эпох и размер батча должны быть положительными.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Скорость обучения должна быть положительным конечным числом.")
    if args.seed < 0:
        parser.error("--seed должен быть неотрицательным.")
    if args.limit_train is not None and not 1 <= args.limit_train <= 55_000:
        parser.error("--limit-train должен быть в диапазоне от 1 до 55000.")
    if args.run_dir.exists():
        parser.error(f"Каталог {args.run_dir} уже существует. Укажите новый --run-dir.")

    try:
        run(args)
    except (NotImplementedError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
