/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.records.reader.impl;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.assertEquals;

public class TestConcatenatingRecordReader {

    @Test
    public void test() throws Exception {

        CSVRecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("datavec-api/iris.dat").getFile()));

        CSVRecordReader rr2 = new CSVRecordReader(0, ',');
        rr2.initialize(new FileSplit(new ClassPathResource("datavec-api/iris.dat").getFile()));

        RecordReader rrC = new ConcatenatingRecordReader(rr, rr2);

        int count = 0;
        while(rrC.hasNext()){
            rrC.next();
            count++;
        }

        assertEquals(300, count);
    }
}
