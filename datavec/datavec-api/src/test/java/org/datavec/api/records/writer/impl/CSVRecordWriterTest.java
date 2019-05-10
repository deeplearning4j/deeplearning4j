/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.records.writer.impl;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class CSVRecordWriterTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testWrite() throws Exception {
        File tempFile = File.createTempFile("datavec", "writer");
        tempFile.deleteOnExit();
        FileSplit fileSplit = new FileSplit(tempFile);
        CSVRecordWriter writer = new CSVRecordWriter();
        writer.initialize(fileSplit,new NumberOfRecordsPartitioner());
        List<Writable> collection = new ArrayList<>();
        collection.add(new Text("12"));
        collection.add(new Text("13"));
        collection.add(new Text("14"));

        writer.write(collection);

        CSVRecordReader reader = new CSVRecordReader(0);
        reader.initialize(new FileSplit(tempFile));
        int cnt = 0;
        while (reader.hasNext()) {
            List<Writable> line = new ArrayList<>(reader.next());
            assertEquals(3, line.size());

            assertEquals(12, line.get(0).toInt());
            assertEquals(13, line.get(1).toInt());
            assertEquals(14, line.get(2).toInt());
            cnt++;
        }
        assertEquals(1, cnt);
    }
}
