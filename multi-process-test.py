import time
import multiprocessing


def do():
    print('---start---')
    time.sleep(1)
    print('---done---')


processes = []

for _ in range(10):
    p = multiprocessing.Process(target=do)
    p.start()
    processes.append(p)

for i in range(len(processes)):
    processes[i].join()
