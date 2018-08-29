################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

 
from collections import OrderedDict
from .conditions import *
from .schema import Schema

def _dq(x):
    return "\"" + x.replace("\"", "\\\"") + "\""

def _to_camel(x, first_upper=False):
    tokens = x.split('_')
    if first_upper:
        y = ''
        for t in tokens:
            y += t[0].upper() + t[1:]
    else:
        y=tokens[0]
        for t in tokens[1:]:
            y += t[0].upper() + t[1:]
    return y


def _dict_to_jmap(d, JMap):
    jmap = JMap()
    for k, v in d.items():
        jmap.put(k, v)
    return jmap


class TransformProcess(object):

    def __init__(self, schema):
        self.schema = schema
        self.final_schema = schema.copy()
        self.steps = []

    def add_step(self, step, *args):
        self.steps.append((step,) + args)

    def remove_column(self, *columns):
        if len(columns) == 1:
            columns = columns[0]
            if type(columns) in (list, tuple):
                self.add_step("removeColumns", *columns)
                for c in columns:
                    del self.final_schema.columns[c]
            else:
                self.add_step("removeColumns", columns)
                del self.final_schema.columns[columns]
        else:
            self.add_step("removeColumns", *columns)
            for c in columns:
                del self.final_schema.columns[c]

    def remove_columns_except(self, *columns):
        if len(columns) == 1:
            columns = columns[0]
            if type(columns) in (list, tuple):
                self.add_step("removeAllColumnsExceptFor", *columns)
                for c in self.final_schema.columns:
                    if c not in columns:
                        del self.final_schema.columns[c]
            else:
                self.add_step("removeAllColumnsExceptFor", columns)
                for c in self.final_schema.columns:
                    if c != columns:
                        del self.final_schema.columns[columns]
        else:
            self.add_step("removeAllColumnsExceptFor", *columns)
            for c in self.final_schema.columns:
                if c not in columns:
                    del self.final_schema.columns[c]

    def filter(self, condition):
        col_name = condition.column
        col_type = self.final_schema.get_column_type(col_name)
        col_type = col_type[0].upper() + col_type[1:]
        if condition.name in ("InSet", "NotInSet"):
            code = "filter(ConditionFilter({}ColumnCondition({}, ConditionOp.{}, HashSet(Arrays.asList({})))))"
            code  = code.format(col_type, _dq(col_name), condition.name, ','.join([_dq(x) for x in condition.set]))
        else:
            code = "filter(ConditionFilter({}ColumnCondition({}, ConditionOp.{}, {})"
            code = code.format(col_type, _dq(col_name), condition.name, condition.value)
        self.add_step("exec", code)


    def replace(self, column, value, condition):
        # there are 2 columns involved
        # the column whose content we are replacing
        # and the column against which the condition is written
        column1_type = self.final_schema.get_column_type(column)
        column1_type = column1_type[0].upper() + column1_type[1:]
        column2 = condition.column
        column2_type = self.final_schema.get_column_type(column2)
        column2_type = column2_type[0].upper() + column2_type[1:]
        if condition.name in ("InSet", "NotInSet"):
            code = "conditionalReplaceValueTransform({}, {}Writable({}), {}ColumnCondition({}, ConditionOp.{}, HashSet(Arrays.asList({}))))"
            code = code.format(_dq(column), column1_type, value, column2_type, _dq(column2), condition.name, ','.join([_dq(x) for x in condition.set]))
        else:
            code = "conditionalReplaceValueTransform({}, {}Writable({}), {}ColumnCondition({}, ConditionOp.{}, {}))"
            code = code.format(_dq(column), column1_type, value, column2_type, _dq(column2), condition.name, condition.value)
        self.add_step("exec", code)

    def rename_column(self, column, new_name):
        new_d = OrderedDict()
        old_d = self.final_schema.columns
        for k in old_d:
            if k == column:
                new_d[new_name] = old_d[k]
            else:
                new_d[k] = old_d[k]
        self.final_schema.columns = new_d
        self.add_step("renameColumn", column, new_name)

    def string_to_time(self, column, format="YYY-MM-DD HH:mm:ss.SSS", time_zone="UTC"):
        self.final_schema.columns[column][0] = "DateTime"
        self.add_step("exec", "stringToTimeTransform({}, {}, {})".format(_dq(column), _dq(format), "DateTimeZone." + time_zone))

    def derive_column_from_time(self, source_column, new_column, field):
        code = 'transform(DeriveColumnsFromTimeTransformBuilder({}).addIntegerDerivedColumn({}, DateTimeFieldType.{}()).build())'
        code = code.format(_dq(source_column), _dq(new_column), _to_camel(field))
        self.add_step("exec", code)
        self.final_schema.add_column("integer", new_column)

    def categorical_to_integer(self, column):
        if self.final_schema.columns[column][0] != 'categorical':
            raise Exception('Can not apply categorical_to_integer'
            ' transform on column \"{}\" because it is not a categorcal column.'.format(column))
        self.final_schema.columns[column][0] = 'integer'
        self.add_step('categoricalToInteger', column)

    def append_string(self, column, string):
        if self.final_schema.columns[column][0] != 'string':
            raise Exception('Can not apply append_string transform to column {} because it is not a string column'.format(column))
        self.add_step('appendStringColumnTransform', column, string)

    def lower(self, column):
        if self.final_schema.columns[column][0] != 'string':
            raise Exception('Can not apply lower transform to column {} because it is not a string column'.format(column))
        self.add_step('exec', 'transform(ChangeCaseStringTransform({}, ChangeCaseStringTransformCaseType.LOWER))'.format(_dq(column)))
        
    def upper(self, column):
        if self.final_schema.columns[column][0] != 'string':
            raise Exception('Can not apply lower transform to column {} because it is not a string column'.format(column))
        self.add_step('exec', 'transform(ChangeCaseStringTransform({}, ChangeCaseStringTransformCaseType.UPPER))'.format(_dq(column)))

    def concat(self, columns, new_column=None, delimiter=','):
        for column in columns:
            if self.final_schema.columns[column][0] != 'string':
                raise Exception('Can not apply concat transform to column {} because it is not a string column'.format(column))
        if new_column is None:
            new_column = 'concat({})'.format(','.join(columns))
        if new_column in self.final_schema.columns:
            raise Exception('Another column with name {} already exists.'.format(new_column))
        columns = [_dq(c) for c in columns]
        self.final_schema.add_string_column(new_column)
        self.add_step('exec', 'transform(ConcatenateStringColumns({}, {}, Arrays.asList({})))'.format(_dq(new_column), _dq(delimiter), ', '.join(columns)))  

    def remove_white_spaces(self, column):
        if self.final_schema.columns[column][0] != 'string':
            raise Exception('Can not apply remove_white_spaces transform to column {} because it is not a string column'.format(column))
        self.add_step('exec', 'transform(RemoveWhiteSpaceTransform({}))'.format(_dq(column)))

    def replace_empty_string(self, column, value):
        if self.final_schema.columns[column][0] != 'string':
            raise Exception('Can not apply replace_empty_string transform to column {} because it is not a string column'.format(column))
        self.add_step('exec', 'transform(ReplaceEmptyStringTransform({}, {}))'.format(_dq(column), _dq(value)))
 
    def replace_string(self, column, *args):
        if self.final_schema.columns[column][0] != 'string':
            raise Exception('Can not apply replace_string transform to column {} because it is not a string column'.format(column))
        if len(args) == 1:
            args = args[0]
            assert type(args) is dict, 'Invalid argument. Possible signatures are replace(str, str, str) and replace(str, dict)'
        elif len(args) == 2:
            assert type(args[0]) == str and type(args[1]) == str, 'Invalid argument. Possible signatures are replace(str, str, str) and replace(str, dict)'
            args = {args[0] : args[1]}
        else:
            raise Exception('Invalid argument. Possible signatures are replace(str, str, str) and replace(str, dict)')
        self.add_step('exec', 'transform(ReplaceStringTransform({}, _dict_to_jmap({}, JMap)))'.format(_dq(column), str(args)))

    def map_string(self, column, mapping):
        if self.final_schema.columns[column][0] != 'string':
            raise Exception('Can not apply replace_string transform to column {} because it is not a string column'.format(column))
        self.add_step('exec', 'transform(StringMapTransform({}, _dict_to_jmap({}, JMap)))'.format(_dq(column), str(mapping)))

    def one_hot(self, column):
        if self.final_schema.columns[column][0] != 'categorical':
            raise Exception('Can not apply one_hot transform to column {} because it is not a categorical column'.format(column))
        categories = self.final_schema.columns[column][2:]
        new_col_names = [column + '[{}]'.format(cat) for cat in categories]
        new_schema = OrderedDict()
        for k in self.final_schema.columns:
            if k == column:
                for c in new_col_names:
                    new_schema[c] = ['integer']
            else:
                new_schema[k] = self.final_schema.columns[k]
        self.final_schema.columns = new_schema
        self.add_step('categoricalToOneHot', column)

    def reduce(self, key, *args, **kwargs):
        # possible signatures:
        # tp.reduce(column_name, default_redcution)  # example: tp.reduce('person', 'sum')  # sums all columns
        # tp.reduce(column, {'amount' : 'sum', 'hours' : 'mean'}) # Explicit reduction for each columns
        # tp.reduce(column, 'sum', {'hours' : 'mean'}) # Explicit reduction for some columns, default reduction for others
        # tp.reduce(column, 'sum', 'hours'='mean')  # kwargs instead of dict
        if type(key) is str:
            key = [key]
        else:
            key = list(key)
        non_key_columns = [x for x in self.final_schema.columns if x not in key]
        col_2_reduction = {}
        if args:
            if type(args[0]) is dict:
                default = None
                col_2_reduction = args[0]
            else:
                default = args[0]
                if len(args) > 1:
                    assert type(args[1]) == dict, 'Expected dict'
                    col_2_reduction = args[1]
                else:
                    col_2_reduction = kwargs
        else:
            default = None
            col_2_reduction = kwargs
        reductions = ['min', 'max', 'sum', 'prod', 'mean', 'std', 'uncorrected_std', 
        'var', 'pop_var', 'count', 'range', 'count_unique','first', 'last', 
        'append', 'prepend']
        if default is None:
            for k in non_key_columns:
                assert k in col_2_reduction, "Reduction not specified for column {}.".format(k)
        else:
            assert default in reductions, "Invalid default reduction {}. Valid redcutions are {}.".format(default, reductions)
        for k, v in col_2_reduction.items():
            assert v in reductions, "Invalid redcution {} specified for column {}. Valid reductions are {}.".format(v, k, reductions)
        reduction_to_function = {'std': 'stddevColumns', 'uncorrected_std' : 'uncorrectedStdevColumns', 'var' : 'variance',
        'pop_var': 'populationVariance', 'first': 'takeFirstColumns', 'last': 'takeLastColumns', 'max' : 'maxColumn'}
        if default is None:
            default = col_2_reduction[col_2_reduction.keys()[0]]
        reduction_to_op = {'std' : 'Stdev', 'uncorrected_std': 'UncorrectedStdDev', 'var': 'Variance', 'pop_var': 'PopulationVariance', 
        'first' : 'TakeFirst', 'last': 'TakeLast'}
        default_op = reduction_to_op.get(default, _to_camel(default, True))
        col_2_function = {}
        for k, v in col_2_reduction.items():
            f = reduction_to_function.get(v, _to_camel(v) + 'Columns')
            col_2_function[k] = f
        code = 'reduce(ReducerBuilder(ReduceOp.{}).keyColumns({})'.format(default_op, ','.join([_dq(k) for k in key]))
        for c, f in col_2_function.items():
            code += ".{}({})".format(f, _dq(c))
        code += '.build())'
        self.add_step('exec', code)
        reduction_to_type = {}
        for r in ['mean', 'std', 'var', 'pop_var', 'uncorrected_std']:
            reduction_to_type[r] = 'double'
        for r in ['append', 'prepend']:
            reduction_to_type[r] = 'string'
        for r in ['count', 'count_unique']:
            reduction_to_type[r] = 'long'
        new_schema = OrderedDict()
        for k, v in self.final_schema.columns.items():
            if k in key:
                new_schema[k] = v
            else:
                reduction = col_2_reduction.get(k, default)
                old_type = v[0]
                op = reduction_to_op.get(reduction, _to_camel(default, True))
                new_name = op.lower() + '(' + k + ')'
                new_type = reduction_to_type.get(reduction, old_type)
                new_schema[k] = [new_type, new_name]
        self.final_schema.columns = new_schema
                
    def serialize(self):
        config = {'steps' : self.steps, 'schema' : self.schema.serialize()}
        return config

    @classmethod
    def deserialize(cls, config):
        schema = Schema.deserialize(config['schema'])
        tp = cls(schema)
        tp.steps = config['steps'][:]
        return tp

    def to_java(self):
        from .java_classes import TransformProcessBuilder
        from .java_classes import ConditionOp
        from .java_classes import ConditionFilter
        from .java_classes import BooleanColumnCondition
        from .java_classes import CategoricalColumnCondition
        from .java_classes import DoubleColumnCondition
        #from .java_classes import FloatColumnCondition
        from .java_classes import StringColumnCondition
        from .java_classes import DateTimeZone
        from .java_classes import DeriveColumnsFromTimeTransformBuilder
        from .java_classes import Arrays, HashSet
        from .java_classes import BooleanWritable
        from .java_classes import IntegerWritable
        from .java_classes import LongWritable
        from .java_classes import FloatWritable
        from .java_classes import DoubleWritable
        from .java_classes import DateTimeFieldType
        from .java_classes import ChangeCaseStringTransform
        from .java_classes import ChangeCaseStringTransformCaseType
        from .java_classes import ConcatenateStringColumns
        from .java_classes import RemoveWhiteSpaceTransform
        from .java_classes import ReplaceEmptyStringTransform
        from .java_classes import ReplaceStringTransform
        from .java_classes import StringMapTransform
        from .java_classes import JMap
        from .java_classes import Arrays
        from .java_classes import ReducerBuilder
        from .java_classes import ReduceOp


        jschema = self.schema.to_java()
        builder = TransformProcessBuilder(jschema)
        for step in self.steps:
            if step[0] == "exec":
                code = step[1]
                print(code)
                exec("builder." + code)
            else:
                f = getattr(builder, step[0])
                f(*step[1:])
        return builder.build()

    def __call__(self, csv):
        try:
            executor = self.executor
        except AttributeError:
            from .executors import SparkExecutor
            executor = SparkExecutor()
            self.executor = executor
        return executor(self, csv)
