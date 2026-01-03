import time
import functools


def execution_timer(func):
    """
    Décorateur qui mesure et affiche le temps d'exécution d'une fonction
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # plus précis que time.time()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f"### >>> Fonction '{func.__name__}' exécutée en {elapsed_time:.4f} secondes  <<< ###")

        return result

    return wrapper