/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * LineRecordReaderFunction: Used to map a {@code JavaRDD<String>} to a {@code JavaRDD<Collection<Writable>>}
 * Note that this is most useful with LineRecordReader instances (CSVRecordReader, SVMLightRecordReader, etc)
 *
 * @author Alex Black
 */
public class LineRecordReaderFunction implements Function<String, List<Writable>> {
    private final RecordReader recordReader;

    public LineRecordReaderFunction(RecordReader recordReader) {
        this.recordReader = recordReader;
    }

    @Override
    public List<Writable> call(String s) throws Exception {
        recordReader.initialize(new StringSplit(s));
        return recordReader.next();
    }
}
