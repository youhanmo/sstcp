from .libs import *


def get_apfd(fails: list, sorted_test_cases: list) -> float:
    # Key point: index can be used to represent a testcase.
    # <============= Bug ===============>
    tfs = [sorted_test_cases.index(fail) + 1 for fail in fails]
    # <============= Bug ===============>

    # tfs = [sorted_test_cases[fail] + 1 for fail in fails]
    n_fail, n_testcase = len(fails), len(sorted_test_cases)
    return 1 - (sum(tfs) / (n_fail * n_testcase)) \
           + 1 / (2 * n_testcase)