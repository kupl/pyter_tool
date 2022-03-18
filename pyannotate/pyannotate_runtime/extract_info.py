# -*- coding: utf-8 -*-
import ast
import sys

class ExtractConstant(ast.NodeVisitor) :
    is_target_func = False
    constant_dict = dict()
    def __init__(self, funcname) :
        self.funcname = funcname

    def visit_Constant(self, node) :
        if self.is_target_func :
            typ = type(node.value).__name__
            typ_list = self.constant_dict.get(typ, [])

            if isinstance(node.value, bytes) :
                typ_list.append(node.value.decode('utf-8'))
            else :
                typ_list.append(node.value)
            self.constant_dict[typ] = list(set(typ_list))

    def visit_Num(self, node) :
        if self.is_target_func :
            typ = type(node.n).__name__
            if typ == 'complex' :
                return
            typ_list = self.constant_dict.get(typ, [])
            typ_list.append(node.n)
            self.constant_dict[typ] = list(set(typ_list))

    def visit_Str(self, node) :
        if self.is_target_func :
            typ = type(node.s).__name__
            typ_list = self.constant_dict.get(typ, [])
            typ_list.append(node.s)
            self.constant_dict[typ] = list(set(typ_list))


    def visit_FunctionDef(self, node) :
        if node.name == self.funcname :
            self.is_target_func = True
        
        self.generic_visit(node)

        self.is_target_func = False

    def visit_AsyncFunctionDef(self, node) :
        if node.name == self.funcname :
            self.is_target_func = True
        
        self.generic_visit(node)

        self.is_target_func = False

    def get_constant(self, node) :
        self.visit(node)

        return self.constant_dict

class ExtractRaise(ast.NodeVisitor) :
    in_with = False
    is_target_with = False
    error_list = []
    msg_list = []
    def __init__(self, funcname) :
        self.funcname = funcname
        self.is_target_func = False

    def visit_With(self, node) :
        ast.NodeVisitor.generic_visit(self, node)

        if self.is_target_func :
            if sys.version_info[0] >= 3 :
                for item in node.items :
                    expr=item.context_expr
                    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name) :
                        # pytest.raises이냐를 확인
                        if expr.func.value.id == 'pytest' and expr.func.attr == 'raises' :
                            args = expr.args
                            keywords = expr.keywords

                            if isinstance(args[0], ast.Name) :
                                error = args[0].id
                                msg = ""
                                if keywords :
                                    if isinstance(keywords[0].value, ast.Constant) :
                                        msg = keywords[0].value.value

                                self.error_list.append(error)
                                self.msg_list.append(msg)
            else :
                expr=node.context_expr
                if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name) :
                    # pytest.raises이냐를 확인
                    if expr.func.value.id == 'pytest' and expr.func.attr == 'raises' :
                        args = expr.args
                        keywords = expr.keywords

                        if isinstance(args[0], ast.Name) :
                            error = args[0].id
                            msg = ""
                            if keywords :
                                if isinstance(keywords[0].value, ast.Constant) :
                                    msg = keywords[0].value.value

                            self.error_list.append(error)
                            self.msg_list.append(msg)


    def generic_visit(self, node) :
        ast.NodeVisitor.generic_visit(self, node)
        '''
        if self.in_with and hasattr(node, 'lineno') and ((not self.lineno) or node.lineno >= self.lineno) :
            self.is_target_with = True
        else :
            self.is_target_with = False
        '''

    def visit_FunctionDef(self, node) :
        if node.name == self.funcname :
            self.is_target_func = True
        
        self.generic_visit(node)

        self.is_target_func = False

    def visit_AsyncFunctionDef(self, node) :
        if node.name == self.funcname :
            self.is_target_func = True
        
        self.generic_visit(node)

        self.is_target_func = False

    def get_raise_info(self, node) :
        self.visit(node)

        return self.error_list, self.msg_list
            