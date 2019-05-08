import pytest
from pydatavec import Schema, TransformProcess


def test_reduce_1():
    reductions = ['sum', 'mean', 'std', 'var', 'prod']
    for red in reductions:
        schema = Schema()
        schema.add_string_column('name')
        schema.add_double_column('amount')
        schema.add_integer_column('hours')

        tp = TransformProcess(schema)
        tp.reduce('name', red)

        tp.to_java()


def test_reduce_2():
    reductions = ['sum', 'mean', 'std', 'var', 'prod']
    for red1 in reductions:
        for red2 in reductions:
            schema = Schema()
            schema.add_string_column('name')
            schema.add_double_column('amount')
            schema.add_integer_column('hours')

            tp = TransformProcess(schema)
            tp.reduce('name', red1, {'amount': red2})

            tp.to_java()


def test_reduce_3():
    reductions = ['sum', 'mean', 'std', 'var', 'prod']
    for red1 in reductions:
        for red2 in reductions:
            schema = Schema()
            schema.add_string_column('name')
            schema.add_double_column('amount')
            schema.add_integer_column('hours')

            tp = TransformProcess(schema)
            tp.reduce('name', {'amount': red1, 'hours': red2})

            tp.to_java()


def test_reduce_4():
    reductions = ['first', 'last', 'append', 'prepend', 'count', 'count_unique']
    for red in reductions:
        schema = Schema()
        schema.add_string_column('col1')
        schema.add_string_column('col2')

        tp = TransformProcess(schema)
        tp.reduce('col1', red)

        tp.to_java()


def test_reduce_5():
    reductions = ['first', 'last', 'append', 'prepend', 'count', 'count_unique']
    for red1 in reductions:
        for red2 in reductions:
            schema = Schema()
            schema.add_string_column('col1')
            schema.add_string_column('col2')
            schema.add_string_column('col3')

            tp = TransformProcess(schema)
            tp.reduce('col1', red1, {'col3': red2})
            tp.to_java()


def test_reduce_6():
    reductions = ['first', 'last', 'append', 'prepend', 'count', 'count_unique']
    for red1 in reductions:
        for red2 in reductions:
            schema = Schema()
            schema.add_string_column('col1')
            schema.add_string_column('col2')
            schema.add_string_column('col3')

            tp = TransformProcess(schema)
            tp.reduce('col1', {'col2': red1, 'col3': red2})

            tp.to_java()


if __name__ == '__main__':
        pytest.main([__file__])
