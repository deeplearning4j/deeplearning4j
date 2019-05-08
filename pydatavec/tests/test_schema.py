import pytest
from pydatavec import Schema


def test_schema():
    schema = Schema()
    schema.add_string_column('str1')
    schema.add_string_column('str2')
    schema.add_integer_column('int1')
    schema.add_integer_column('int2')
    schema.add_double_column('dbl1')
    schema.add_double_column('dbl2')
    schema.add_float_column('flt1')
    schema.add_float_column('flt2')
    schema.add_categorical_column('cat1', ['A', 'B', 'C'])
    schema.add_categorical_column('cat2', ['A', 'B', 'C'])
    schema.to_java()


if __name__ == '__main__':
    pytest.main([__file__])
