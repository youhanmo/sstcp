from .libs import *


def tokenize(code: str) -> list:
    """
    Parameters:
    -----------
    - code: str, a segment(or statement) of code.

    Returns:
    --------
    - words: list, like ["this", "is", "word"]
    """
    # Remove some special charactor.
    special_word = [',','/','+',')','.','(',';','{','}','<','>','"','"', '\'','=']
    for sub in special_word:
        code = code.replace(sub,' ')
    code = code.replace('\t',' ')
    code = code.replace('\n',' ')
    code = ' '.join(code.strip().split())
    term_list = []
    term_list = code.split(" ")
    lower_term = []
    for word in term_list:
        if len(word.strip()) != 0:
            new_word = word.strip()
            if len(new_word) > 1:
                for sub_word in re.sub(r"([A-Z])", r" \1",new_word).split():
                    lower_term.append(sub_word.lower())
            else:
                lower_term.append(new_word)
    return lower_term