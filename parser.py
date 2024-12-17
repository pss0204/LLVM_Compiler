import ply.yacc as yacc
from lexer import tokens

class ASTNode:
    pass

class Assign(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class BinOp(ASTNode):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class Number(ASTNode):
    def __init__(self, value):
        self.value = value

def p_statement_assign(p):
    'statement : IDENT EQUALS expression'
    p[0] = Assign(p[1], p[3])

def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression TIMES expression'''
    p[0] = BinOp(p[2], p[1], p[3])

def p_expression_number(p):
    'expression : NUMBER'
    p[0] = Number(p[1])

def p_expression_ident(p):
    'expression : IDENT'
    p[0] = p[1]

def p_error(p):
    print("Syntax error")

parser = yacc.yacc()