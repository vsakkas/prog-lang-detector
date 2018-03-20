import json

languages = ('Ada', 'AppleScript', 'AWK', 'BBC-BASIC', 'C', 'C++', 'C-sharp', 'Clojure', 'COBOL', 'Common-Lisp',
             'D', 'Elixir', 'Erlang', 'Forth', 'Fortran', 'Go', 'Groovy', 'Haskell', 'Icon', 'J', 'Java',
             'JavaScript', 'Julia', 'Kotlin', 'LiveCode', 'Lua', 'Maple', 'MATLAB', 'Objective-C', 'OCaml', 'Oz',
             'Perl', 'PHP', 'PL-I', 'PowerShell', 'Prolog', 'Python', 'R', 'Racket', 'REXX', 'Ring', 'Ruby',
             'Rust', 'Scala', 'Scheme', 'Swift', 'Tcl', 'UNIX-Shell', 'VBScript', 'Visual-Basic-.NET')


def load_keywords():
    """
    Load `keywords.json` and convert it to a dictionary.
    Returns
    ----------
    dict
    """
    with open('keywords.json', 'r') as fd:
        data = fd.read()
        keywords = json.loads(data)
        return keywords


def strip_non_keywords(data, keywords, language):
    """
    Removes alphanumerics that are not in the keywords list within the provided dictionary of keywords per language.
    Parameters
    ----------
    data : string
        The content of the source file from which all non keywords will be removed.
    keywords : dict
        Dictionary containing keywords per language.
    language : string
        Name of the language of the provided file. Used to find the language's keywords in the `keywords` dictionary.
    Returns
    ----------
    string
        The same string as `data` but with all non keywords removed.
    Notes
    ----------
    If the list of keywords in the provided dictionary is empty, then no alphanumerics are removed at all and the
    same string is returned as the one provided.
    """
    if (len(keywords[language]) == 0):
        return data
    token = ""
    i = 0
    while i < len(data):
        if data[i].isalnum():
            token += data[i]
            i += 1
            continue
        if token.isalnum() and not token.isnumeric() and token not in keywords[language]:
            data = 'w'.join((data[:i - len(token)], data[i:]))
            i += len('w') - len(token)
        token = ""
        i += 1
    return data
