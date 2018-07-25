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

package org.datavec.api.records.reader.impl;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVLineSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class CSVLineSequenceRecordReaderTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void test() throws Exception {

        File f = testDir.newFolder();
        File source = new File(f, "temp.csv");
        String str = "a,b,c\n1,2,3,4";
        FileUtils.writeStringToFile(source, str);

        SequenceRecordReader rr = new CSVLineSequenceRecordReader();
        rr.initialize(new FileSplit(source));

        List<List<Writable>> exp0 = Arrays.asList(
                Collections.<Writable>singletonList(new Text("a")),
                Collections.<Writable>singletonList(new Text("b")),
                Collections.<Writable>singletonList(new Text("c")));

        List<List<Writable>> exp1 = Arrays.asList(
                Collections.<Writable>singletonList(new Text("1")),
                Collections.<Writable>singletonList(new Text("2")),
                Collections.<Writable>singletonList(new Text("3")),
                Collections.<Writable>singletonList(new Text("4")));

        for( int i=0; i<3; i++ ) {
            int count = 0;
            while (rr.hasNext()) {
                List<List<Writable>> next = rr.sequenceRecord();
                if (count++ == 0) {
                    assertEquals(exp0, next);
                } else {
                    assertEquals(exp1, next);
                }
            }

            assertEquals(2, count);

            rr.reset();
        }
    }

}
