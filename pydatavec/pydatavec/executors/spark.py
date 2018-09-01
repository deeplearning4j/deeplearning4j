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


import os

_JVM_RUNNING = False


class StringRDD(object):

    def __init__(self, java_rdd):
        self.java_rdd = java_rdd

    def __iter__(self):
        jlist = self.java_rdd.collect()
        size = jlist.size()
        return iter([jlist.get(i) for i in range(size)])

    def save(self, path):
        self.java_rdd.saveAsTextFile(path)

    def save_to_csv(self, path):
        l = list(self)
        with open(path, 'w') as f:
            for x in l:
                f.write(x + '\n')


class SparkExecutor(object):

    def __init__(self, master='local[*]', app_name='pydatavec'):
        global _JVM_RUNNING
        if not _JVM_RUNNING:
            from ..java_classes import SparkConf, SparkContext, SparkTransformExecutor
            from ..java_classes import CSVRecordReader, WritablesToStringFunction, StringToWritablesFunction
            _JVM_RUNNING = True
        spark_conf = SparkConf()
        spark_conf.setMaster(master)
        spark_conf.setAppName(app_name)
        self.spark_context = SparkContext(spark_conf)
        self.rr = CSVRecordReader()
        self.executor = SparkTransformExecutor
        self.str2wf = StringToWritablesFunction
        self.w2strf = WritablesToStringFunction

    def __call__(self, tp, source):
        source_type = getattr(type(source), '__name__', None)
        if source_type == 'str':
            if os.path.isfile(source) or os.path.isdir(source):
                string_data = self.spark_context.textFile(source)  # JavaRDD<String>
            else:
                raise ValueError('Invalid source ' + source)
        elif source_type == 'org.apache.spark.api.java.JavaRDD':
            string_data = source
        elif source_type.endswith('RDD'):
            tempid = 0
            path = 'temp_0'
            while(os.path.isdir(path)):
                tempid += 1
                path = 'temp_' + str(tempid)
            print('Converting pyspark RDD to JavaRDD...')
            source.saveAsTextFile(path)
            string_data = self.spark_context.textFile(path)
        else:
            raise Exception('Unexpected source type: ' + str(type(source)))
        parsed_input_data = string_data.map(self.str2wf(self.rr))   # JavaRDD<List<Writable>>
        processed_data = self.executor.execute(parsed_input_data, tp.to_java())  # JavaRDD<List<Writable>>
        processed_as_string = processed_data.map(self.w2strf(","))  # JavaRDD<String>
        return StringRDD(processed_as_string)  # StringRDD
