# -*- coding: utf-8 -*-

import dis
import ast

class VarAnalysis(ast.NodeVisitor) :
    def __init__(self) :
        self.line = None
        self.range = None

        self.func_subs = dict([])
        self.func_attr = dict([])
        self.func_var = set([])

    def visit_Subscript(self, node) :
        self.generic_visit(node)
        if self.range and (self.range[0] <= node.lineno <= self.range[1]):
            key = None
            value = None
            if isinstance(node.value, ast.Name) :
                key = node.value.id
            if isinstance(node.value, ast.Attribute) :
                try :
                    key = node.value.value.id + '.'  + node.value.attr
                except :
                    pass
            target_slice = node.slice
            if isinstance(node.slice, ast.Index) :
                target_slice = node.slice.value
            if isinstance(target_slice, ast.Name) :
                value = target_slice.id
            if hasattr(ast, 'Constant') and isinstance(target_slice, ast.Constant) :
                value = target_slice.value
            if isinstance(target_slice, ast.Str) :
                value = target_slice.s
            if isinstance(target_slice, ast.Num) :
                value = target_slice.n
            if key is not None and value is not None :
                self.func_subs[key] = value

    def visit_Attribute(self, node) :
        self.generic_visit(node)
        if self.range and (self.range[0] <= node.lineno <= self.range[1]):
            attr_list = self.func_attr.get(node.value, list())
            attr_list.append(node)

            self.func_attr[node.value] = attr_list

    def visit_Name(self, node) :
        self.generic_visit(node)
        if self.range and (self.range[0] <= node.lineno <= self.range[1]):
            self.func_var.add(node)


    def generic_visit(self, node) :
        if isinstance(node, ast.expr) :
            if self.range :
                if not (self.range[0] <= node.lineno <= self.range[1]) :
                    return
            else :
                if hasattr(node, "end_lineno") :
                    if node.lineno <= self.line <= node.end_lineno :
                        #print(node)
                        self.range = (node.lineno, node.end_lineno)

                else :
                    #print(node.lineno, node)
                    if node.lineno == self.line :
                        #print(node)
                        self.range = (node.lineno, node.lineno)

        ast.NodeVisitor.generic_visit(self, node)

    def find_error_stmt(self, node, lineno) :
        '''
        에러가 난 node 중 가장 작은 단위의 stmt를 구하자
        '''
        smaller_stmt = None 

        for child in ast.iter_child_nodes(node) :
            if isinstance(child, ast.stmt) and hasattr(child, "lineno") :
                if hasattr(node, "end_lineno") :
                    if child.lineno <= lineno <= child.end_lineno :
                        smaller_stmt = self.find_error_stmt(child, lineno)
                        if smaller_stmt is None :
                            return child

                        return smaller_stmt
                else :
                    if child.lineno <= lineno :
                        smaller_stmt = self.find_error_stmt(child, lineno)

                        if smaller_stmt is None :
                            smaller_stmt = child
                
        return smaller_stmt

    def get_usage_var(self) :
        result = set([]) 
        for var_node in self.func_var :
            def get_attr(node, name, output) :
                if isinstance(node, ast.Name) :
                    name = node.id

                if isinstance(node, ast.Attribute) :
                    name = name + '.' + node.attr


                output.add(name)

                for child in self.func_attr.get(node, []) :
                    get_attr(child, name, output)

                return output

            var_list = get_attr(var_node, '', set([]))
            result = result.union(var_list)

        return result


    def get_var_info(self, node, line) :
        self.line = line
        #print("LINE", line)
        #print(ast.dump(node, include_attributes=True))
        #print(line)
        #error_node = self.find_error_stmt(node, line)
        #print(ast.unparse(node))
        self.visit(node)
        tmp = self.get_usage_var()
      
        return tmp, self.func_subs

