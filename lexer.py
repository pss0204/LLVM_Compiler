import ply.lex as lex

tokens = (
    'NUMBER',
    'IDENT',
    'PLUS',
    'TIMES',
    'EQUALS',
)

t_PLUS = r'\+'
t_TIMES = r'\*'
t_EQUALS = r'='
t_ignore = ' \t'

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_IDENT(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    return t

def t_newline(t):
    r'\n+'
    pass

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()