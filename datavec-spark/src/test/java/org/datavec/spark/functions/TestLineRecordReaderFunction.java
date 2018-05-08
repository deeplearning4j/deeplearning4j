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

import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 21/05/2016.
 */
public class TestLineRecordReaderFunction extends BaseSparkTest {

    @Test
    public void testLineRecordReader() throws Exception {

        File dataFile = new ClassPathResource("iris.dat").getFile();
        List<String> lines = FileUtils.readLines(dataFile);

        JavaSparkContext sc = getContext();
        JavaRDD<String> linesRdd = sc.parallelize(lines);

        CSVRecordReader rr = new CSVRecordReader(0, ',');

        JavaRDD<List<Writable>> out = linesRdd.map(new LineRecordReaderFunction(rr));
        List<List<Writable>> outList = out.collect();


        CSVRecordReader rr2 = new CSVRecordReader(0, ',');
        rr2.initialize(new FileSplit(dataFile));
        Set<List<Writable>> expectedSet = new HashSet<>();
        int totalCount = 0;
        while (rr2.hasNext()) {
            expectedSet.add(rr2.next());
            totalCount++;
        }

        assertEquals(totalCount, outList.size());

        for (List<Writable> line : outList) {
            assertTrue(expectedSet.contains(line));
        }
    }
}
