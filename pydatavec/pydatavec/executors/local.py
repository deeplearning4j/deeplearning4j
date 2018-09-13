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



class Writable(object):

    def __init__(self, j_w):
        self.j_w = j_w

    def save_to_csv(self, path):
        from ..java_classes import NumberOfRecordsPartitioner
        from ..java_classes import CSVRecordWriter
        from ..java_classes import FileSplit, JFile

        output_file = JFile(path)
        if output_file.exists():
            output_file.delete()
        output_file.createNewFile()
        rw = CSVRecordWriter()
        rw.initialize(FileSplit(output_file), NumberOfRecordsPartitioner())
        rw.writeBatch(self.j_w)
        rw.close()

    def save(self, path):
        self.save_to_csv(path)

    def __iter__(self):
        rows = []
        nr = self.j_w.size()
        nc = self.j_w.get(0).size() if nr else 0
        for i in range(nr):
            row = self.j_w.get(i)
            cols = [row.get(j).toString() for j in range(nc)]
            rows.append(cols)
        return iter(rows)

    def iter(self):
        return self.__iter__()


class LocalExecutor(object):

    def __init__(self):
        from ..java_classes import CSVRecordReader
        self.rr = CSVRecordReader(0, ',')

        pass

    def __call__(self, tp, source):
        from ..java_classes import CSVRecordReader, WritablesToStringFunction, StringToWritablesFunction
        from ..java_classes import FileSplit, JFile, ArrayList, LocalTransformExecutor

        tp = tp.to_java()
        assert type(source) is str
        assert os.path.isfile(source)
        f = JFile(source)
        rr = self.rr
        rr.initialize(FileSplit(f))
        data = ArrayList()
        while rr.hasNext():
            data.add(rr.next())
        processed_data = LocalTransformExecutor.execute(data, tp)
        return Writable(processed_data)
