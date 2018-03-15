import json

languages = ('Ada', 'AppleScript', 'AWK', 'BBC-BASIC', 'C', 'C++', 'C-sharp', 'Clojure', 'COBOL', 'Common-Lisp',
             'D', 'Elixir', 'Erlang', 'Forth', 'Fortran', 'Go', 'Groovy', 'Haskell', 'Icon', 'J', 'Java',
             'JavaScript', 'Julia', 'Kotlin', 'LiveCode', 'Lua', 'Maple', 'MATLAB', 'Objective-C', 'OCaml', 'Oz',
             'Perl', 'PHP', 'PL-I', 'PowerShell', 'Prolog', 'Python', 'R', 'Racket', 'REXX', 'Ring', 'Ruby',
             'Rust', 'Scala', 'Scheme', 'Swift', 'Tcl', 'UNIX-Shell', 'VBScript', 'Visual-Basic-.NET')


def load_keywords():
    with open('../keywords.json', 'r') as fd:
        data = fd.read()
        keywords = json.loads(data)
        return keywords


def strip_non_keywords(data, keywords, language):
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
