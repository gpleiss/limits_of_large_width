import fire
import math
from runner import Runner, new
import random


def run(n=5000, width=256, std=20., seed=0, runs=10):
    for run in range(1, runs + 1):
        random.seed(run)
        lr = 10 ** (random.uniform(-3, math.log10(0.2)))
        batch_size = random.choice([16, 32, 64, 128, 256])
        print("Hyperparameters:")
        print(f" - lr: {lr}")
        print(f" - batch_size: {batch_size}")
        print("")
        
        try:
            savedir = f"results/mnist{n}_nn_w{width}+{width}+{width}_s{seed}_std{std}_r{run}"
            runner = new(
                overwrite=True, save=savedir, dataset="mnist", n=n, model="nn", std=std,
                seed=seed, widths=[width, width, width],
            )
            runner.train(lr=lr, minibatch_size=batch_size).test()
        except Exception as e:
            raise e


if __name__ == "__main__":
    fire.Fire(run)
