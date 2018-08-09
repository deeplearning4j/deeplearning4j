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


import jnius_config
import os


class_path = os.environ.get('DATAVEC_CLASS_PATH')

if class_path is None:
    raise Exception('Environment variable DATAVEC_CLASS_PATH not set.')

if class_path[-1] == '/':
    class_path += '*'
else:
    class_path += '/*'


jnius_config.set_classpath(class_path)

from jnius import autoclass


JSchema = autoclass('org.datavec.api.transform.schema.Schema')
SchemaBuilder = autoclass('org/datavec/api/transform/schema/Schema$Builder')

JTransformProcess = autoclass('org.datavec.api.transform.TransformProcess')
TransformProcessBuilder = autoclass('org/datavec/api/transform/TransformProcess$Builder')

ConditionOp = autoclass('org.datavec.api.transform.condition.ConditionOp')
ConditionFilter = autoclass('org.datavec.api.transform.filter.ConditionFilter')

BooleanColumnCondition = autoclass('org.datavec.api.transform.condition.column.BooleanColumnCondition')
CategoricalColumnCondition = autoclass('org.datavec.api.transform.condition.column.CategoricalColumnCondition')
DoubleColumnCondition = autoclass('org.datavec.api.transform.condition.column.DoubleColumnCondition')
#FloatColumnCondition = autoclass('org.datavec.api.transform.condition.column.FloatColumnCondition')
StringColumnCondition = autoclass('org.datavec.api.transform.condition.column.StringColumnCondition')


BooleanWritable = autoclass('org.datavec.api.writable.BooleanWritable')
IntegerWritable = autoclass('org.datavec.api.writable.IntWritable')
LongWritable = autoclass('org.datavec.api.writable.LongWritable')
FloatWritable = autoclass('org.datavec.api.writable.FloatWritable')
DoubleWritable = autoclass('org.datavec.api.writable.DoubleWritable')



DateTimeZone = autoclass('org.joda.time.DateTimeZone')
DateTimeFieldType = autoclass('org.joda.time.DateTimeFieldType')
DeriveColumnsFromTimeTransformBuilder = autoclass('org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform$Builder')


Arrays = autoclass('java.util.Arrays')
HashSet = autoclass('java.util.HashSet')


JDouble = autoclass('java.lang.Double')
JFloat = autoclass('java.lang.Float')
