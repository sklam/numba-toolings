"""
Compute the maximum coverage
"""
import sys
import pickle
from typing import Tuple
from pprint import pprint
from coverage import CoverageData
from collections import defaultdict

import numpy as np
from scipy.sparse import dok_array

def prepare_data(covfilepath: str) -> Tuple[np.array, list[str]]:

    cd = CoverageData(covfilepath)
    cd.read()
    contexts = cd.measured_contexts()
    files = cd.measured_files()
    print(f"{len(files)} files and {len(contexts)} contexts")

    # Map context to index in `contexts`
    context_map: dict[str, int] = {ctx: i for i, ctx in enumerate(contexts)}

    # Map file to index in `files`
    file_map: dict[str, int] = {fn: i for i, fn in enumerate(files)}

    # ContextIdx: {FileIdx: Set[Lines]}
    ctx_file_map: dict[int, dict[int, set[int]]] = {}

    # Line per file
    line_per_file: dict[int, set[int]] = defaultdict(set)

    for fn in files:
        if "numba/tests" in fn:
            # Ignore files in test directory
            print("ignoring", fn)
            continue


        line_ctx_map: dict[int, list[str]] = cd.contexts_by_lineno(fn)
        fidx = file_map[fn]

        for ln, ctxlist in line_ctx_map.items():
            for ctx in filter(bool, ctxlist): # filter to ignore empty context
                where = ctx_file_map.setdefault(context_map[ctx], {})
                per_ctx = where.setdefault(fidx, set())
                per_ctx.add(ln)
            line_per_file[fidx].add(ln)

    # Compute file-line map to vector
    file_line_map: dict[(int, int): int] = {}
    pos = 0
    for fn, lines in line_per_file.items():
        for ln in sorted(lines):
            file_line_map[fn, ln] = pos
            pos += 1
    # pprint(file_line_map)

    if True:
        # Convert to sparse vector
        shape = len(contexts), len(file_line_map)
        print("Matrix shape", shape)
        matrix = dok_array(shape, dtype=np.float32)
    else:
        # Convert to dense vector
        shape = len(contexts), len(file_line_map)
        print("Matrix shape", shape)
        matrix = np.zeros(shape, dtype=np.float32)
        print("Matrix size", matrix.nbytes / 2**20, "MB")

    # Fill the matrix
    for ctx, file_lines in ctx_file_map.items():
        for fn, lines in file_lines.items():
            for ln in lines:
                pos = file_line_map[fn, ln]
                matrix[ctx, pos] = 1

    return matrix, list(contexts)

def dump_solution(soln, contexts: list[str]):
    print("Dump solution".center(40, '-'))
    for i in soln:
        print(contexts[i])

def evaluate(matrix, soln) -> int:
    """
    Given the list of selected tests,
    returns the number of covered lines for the solution.
    """
    # Max across all test to compute the expected coverage
    tally = np.max(matrix[soln], axis=0)
    assert tally.shape[1] == matrix.shape[1], (tally.shape, matrix.shape)
    # Count covered lines.
    nnz = tally.count_nonzero()
    return nnz


def pca_feature_extract(pca):
    # Reference this answer https://stackoverflow.com/a/50845697
    return np.abs(pca.components_).argmax(axis=1)

# def pca_feature_extract(pca):
#     # L1 norm across the principal components seems to get good "importance" weights for each file
#     # print("pca.components_.shape", pca.components_.shape)
#     weights = np.asarray([np.linalg.norm(x, ord=1) for x in pca.components_.T])
#     # print("weights.shape", weights.shape)
#     # Sort the weights and pull out the most "important" files
#     soln = np.argsort(weights)[-pca.components_.shape[0]:]
#     return soln



def main():
    [infile, outfile] = sys.argv[1:]
    dok, contexts = prepare_data(infile)
    csr = dok.tocsr()
    data = (csr, contexts)
    with open(outfile, "wb") as fout:
        pickle.dump(data, fout)

if __name__ == "__main__":
    main()
