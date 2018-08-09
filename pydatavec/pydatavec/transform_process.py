from collections import OrderedDict
from .conditions import *
from .schema import Schema

def _dq(x):
    return "\"" + x.replace("\"", "\\\"") + "\""

def _to_camel(x):
    tokens = x.split('_')
    y=tokens[0]
    for t in tokens[1:]:
        y += t[0].upper() + t[1:]
    return y

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

        jschema = self.schema.to_java()
        builder = TransformProcessBuilder(jschema)
        for step in self.steps:
            if step[0] == "exec":
                code = step[1]
                exec("builder." + code)
            else:
                f = getattr(builder, step[0])
                f(*step[1:])
        return builder.build()

        