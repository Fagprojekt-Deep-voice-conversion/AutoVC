import time
import sys


def progbar(i, N, info = {}):
    # gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
    pbar = _progbar(i, N)
    msg = f'| {pbar} {i}/{N} | '
    for key, val in info.items():
        msg += str(key) + ": " + str(val) + " | "
    stream(msg)

def _progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def stream(message):
    sys.stdout.write(f"\r{message}")


if __name__ == "__main__":
    N = 10000
    for i in range(N):
        progbar(i, N, {"Batchsize" : 10, "Epoch num": 1})