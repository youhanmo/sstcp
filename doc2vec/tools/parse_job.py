from .libs import *
from .tokenize import tokenize


def is_code_change(_dict: dict) -> bool:
    """ Check a dict represents a code change or not. """
    return "diff0" in _dict.keys()


def is_test_case(_dict: dict) -> bool:
    """ Check a dict represents a test case or not. """
    return not is_code_change(_dict)


""" Parse job <without> concatenated methods. <without tokenize> """
def parse_job(job: list) -> tuple:
    # Model inputs.
    test_cases_text = []
    code_changes_text = []

    # Indices of failed test case.
    fails = []

    # Read test cases and code changes.
    test_cases, code_changes = [], []
    for _dict in job:
        if is_test_case(_dict):
            test_cases.append(_dict)
        else:
            code_changes.append(_dict)

    for index, test_case in enumerate(test_cases):
        # Get test cases content, type of content is list.
        test_cases_text.append(test_case.get("content"))

        # Check if this test case failed.
        if test_case.get("fail"):
            fails.append(index)

    for code_change in code_changes:
        code_changes_text.extend(code_change.get("diff0")) # <--- Attention: extend, not append.

    return test_cases_text, code_changes_text, fails