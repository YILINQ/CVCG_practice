import time
import multiprocessing


def do():
    print('---start---')
    time.sleep(1)
    print('---done---')


if __name__ == "__main__":
    processes = []

    start = time.perf_counter()

    for _ in range(10):
        p = multiprocessing.Process(target=do)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    finish = time.perf_counter()

    print(finish - start)
