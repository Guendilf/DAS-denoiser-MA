import multiprocessing
import time
import traceback
import signal
from n2selfDas import main as n2selfDAS_main
from n2noiseDas import main as n2noiseDAS_main
from n2sameDas import main as n2sameDAS_main
from n2infoDas import main as n2infoDAS_main
from import_files import log_files


processes = []

def terminate_processes(signum, frame):
    print("Terminating all child processes...")
    for p in processes:
        p.terminate()
    print("All child processes terminated.")
    exit(0)

def run_process(target, args):
    try:
        target(args)
    except Exception as e:
        print(f"Exception in process {target.__module__}: {e}")
        traceback.print_exc() 

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate_processes)
    signal.signal(signal.SIGINT, terminate_processes)  # Handle Ctrl+C
    # Globalen Speicherpfad abrufen
    global_store_path = log_files()

    # Prozesse erstellen und die `main`-Funktionen mit dem Pfad als Argument übergeben
    p1 = multiprocessing.Process(target=run_process, args=(n2selfDAS_main, [global_store_path]))
    p2 = multiprocessing.Process(target=run_process, args=(n2noiseDAS_main, [global_store_path]))
    p3 = multiprocessing.Process(target=run_process, args=(n2sameDAS_main, [global_store_path]))
    p4 = multiprocessing.Process(target=run_process, args=(n2infoDAS_main, [global_store_path]))

    processes.extend([p1, p2, p3, p4])

    # Prozesse starten (max 6)
    print("Starting process 1 ...")
    p1.start()
    time.sleep(20)
    print("Starting process 2 ...")
    p2.start()
    time.sleep(20)
    print("Starting process 3 ...")
    p3.start()
    time.sleep(20)
    print("Starting process 4 ...")
    p4.start()

    # Warten, bis beide Prozesse abgeschlossen sind
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    print("Both processes have completed.")