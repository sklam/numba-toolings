import multiprocessing as mp
from subprocess import check_output, TimeoutExpired, STDOUT
import traceback
from timeit import default_timer as timer
from functools import partial
import pickle

# A test can't run longer than 10 minutes
_TIMEOUT = 600

def job(fn, use_coverage=True):
    print("running", fn)
    ts = timer()

    prefix = ["coverage", "run", f"--context={fn!r}"] if use_coverage else ["python"]
    try:
        check_output([*prefix, "-m", "unittest", "-vb", f"{fn}"],
                     timeout=_TIMEOUT,
                     stderr=STDOUT)
    except TimeoutExpired:
        print("test timed out!!!", fn)
    except Exception:
        traceback.print_exc()
    else:
        te = timer()
        return fn, te - ts, True
    return fn, None, False


# Run coverage and set the context
def run_coverage(use_coverage=True):
    with open("test_seed.txt", "r") as fin:
        tests = fin.read().splitlines()

    unfinished = set(tests)
    if not use_coverage:
        task = partial(job, use_coverage=False)
    else:
        task = job
    timings = {}
    with mp.Pool(mp.cpu_count()) as pool:
        it = pool.imap_unordered(task, tests)
        while True:
            try:
                fn, elapsed, status = it.__next__(_TIMEOUT + 60)
            except StopIteration:
                break
            except mp.TimeoutError:
                print("A test timedout!!!")
            else:
                if status:
                    unfinished.discard(fn)
                    timings[fn] = elapsed
                    print(fn, "... done")
                else:
                    print(fn, "... failed")

    if unfinished:
        print("Unfinished".center(80, '-'))
        for t in unfinished:
            print(t)

    with open("timings.dat", "wb") as fout:
        pickle.dump(timings, fout)
    print("Ending Gracefully")

