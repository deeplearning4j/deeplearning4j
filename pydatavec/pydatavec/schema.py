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


class Schema(object):

    def __init__(self):
        self.columns = OrderedDict()

    def add_column(self, column_type, column_name, *args):
        if column_name in self.columns:
            raise Exception("Column names should be unique. Another column with name " + column_name + " already exists.")
        self.columns[column_name] = [column_type] + list(args)

    def add_string_column(self, column):
        self.add_column("string", column)

    def add_integer_column(self, column, *args):
        self.add_column("integer", column, *args)

    def add_long_column(self, column, *args):
        self.add_column("long", column, *args)

    def add_float_column(self, column, *args):
        self.add_column("float", column, *args)

    def add_double_column(self, column, *args):
        self.add_column("double", column, *args)

    def add_categorical_column(self, column, categories):
        self.add_column("categorical", column, *categories)

    def get_column_type(self, column):
        return self.columns[column][0]

    def serialize(self):
        config = {}
        meta = []
        col_names = []
        for k in self.columns:
            meta.append(self.columns[k])
            col_names.append(k)
        config['column_names'] = col_names
        config['meta'] = meta
        return config

    @classmethod
    def deserialize(cls, config):
        schema = cls()
        col_names = config['column_names']
        meta = config['meta']
        for c, m in zip(col_names, meta):
            schema.columns[c] = m
        return schema

    def to_java(self):
        from .java_classes import SchemaBuilder
        from .java_classes import JFloat, JDouble
        builder = SchemaBuilder()
        for c in self.columns:
            meta = self.columns[c]
            col_type = meta[0]
            col_name = c
            col_args = meta[1:]
            if col_type == "string":
                builder.addColumnString(col_name)
            elif col_type == "categorical":
                builder.addColumnCategorical(col_name, *col_args)
            else:
                # numerics
                num_type = col_type[0].upper() + col_type[1:]
                f = getattr(builder, 'addColumn' + num_type)
                col_args = list(col_args)
                if num_type in ('Float', 'Double'):
                    jtype = eval('J' + num_type)
                    for i, a in enumerate(col_args):
                        if type(a) in [int, float]:
                            col_args[i] = jtype(a)
                f(col_name, *col_args)
        return builder.build()

    def copy(self):
        config = str(self.serialize())
        clone = Schema.deserialize(eval(config))
        return clone