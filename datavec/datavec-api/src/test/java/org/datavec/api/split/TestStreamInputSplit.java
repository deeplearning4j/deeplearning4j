/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.split;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.function.Function;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class TestStreamInputSplit extends BaseND4JTest {



    @Test
    public void testCsvSimple(@TempDir Path testDir) throws Exception {
        File dir = testDir.toFile();
        File f1 = new File(dir, "file1.txt");
        File f2 = new File(dir, "file2.txt");

        FileUtils.writeStringToFile(f1, "a,b,c\nd,e,f", StandardCharsets.UTF_8);
        FileUtils.writeStringToFile(f2, "1,2,3", StandardCharsets.UTF_8);

        List<URI> uris = Arrays.asList(f1.toURI(), f2.toURI());

        CSVRecordReader rr = new CSVRecordReader();

        TestStreamFunction fn = new TestStreamFunction();
        InputSplit is = new StreamInputSplit(uris, fn);
        rr.initialize(is);

        List<List<Writable>> exp = new ArrayList<>();
        exp.add(Arrays.<Writable>asList(new Text("a"), new Text("b"), new Text("c")));
        exp.add(Arrays.<Writable>asList(new Text("d"), new Text("e"), new Text("f")));
        exp.add(Arrays.<Writable>asList(new Text("1"), new Text("2"), new Text("3")));

        List<List<Writable>> act = new ArrayList<>();
        while(rr.hasNext()){
            act.add(rr.next());
        }

        assertEquals(exp, act);

        //Check that the specified stream loading function was used, not the default:
        assertEquals(uris, fn.calledWithUris);

        rr.reset();
        int count = 0;
        while(rr.hasNext()) {
            count++;
            rr.next();
        }
        assertEquals(3, count);
    }


    @Test
    public void testCsvSequenceSimple(@TempDir Path testDir) throws Exception {

        File dir = testDir.toFile();
        File f1 = new File(dir, "file1.txt");
        File f2 = new File(dir, "file2.txt");

        FileUtils.writeStringToFile(f1, "a,b,c\nd,e,f", StandardCharsets.UTF_8);
        FileUtils.writeStringToFile(f2, "1,2,3", StandardCharsets.UTF_8);

        List<URI> uris = Arrays.asList(f1.toURI(), f2.toURI());

        CSVSequenceRecordReader rr = new CSVSequenceRecordReader();

        TestStreamFunction fn = new TestStreamFunction();
        InputSplit is = new StreamInputSplit(uris, fn);
        rr.initialize(is);

        List<List<List<Writable>>> exp = new ArrayList<>();
        exp.add(Arrays.asList(
                Arrays.<Writable>asList(new Text("a"), new Text("b"), new Text("c")),
                Arrays.<Writable>asList(new Text("d"), new Text("e"), new Text("f"))));
        exp.add(Arrays.asList(
                Arrays.<Writable>asList(new Text("1"), new Text("2"), new Text("3"))));

        List<List<List<Writable>>> act = new ArrayList<>();
        while (rr.hasNext()) {
            act.add(rr.sequenceRecord());
        }

        assertEquals(exp, act);

        //Check that the specified stream loading function was used, not the default:
        assertEquals(uris, fn.calledWithUris);

        rr.reset();
        int count = 0;
        while(rr.hasNext()) {
            count++;
            rr.sequenceRecord();
        }
        assertEquals(2, count);
    }

    @Test
    public void testShuffle(@TempDir Path testDir) throws Exception {
        File dir = testDir.toFile();
        File f1 = new File(dir, "file1.txt");
        File f2 = new File(dir, "file2.txt");
        File f3 = new File(dir, "file3.txt");

        FileUtils.writeStringToFile(f1, "a,b,c", StandardCharsets.UTF_8);
        FileUtils.writeStringToFile(f2, "1,2,3", StandardCharsets.UTF_8);
        FileUtils.writeStringToFile(f3, "x,y,z", StandardCharsets.UTF_8);

        List<URI> uris = Arrays.asList(f1.toURI(), f2.toURI(), f3.toURI());

        CSVSequenceRecordReader rr = new CSVSequenceRecordReader();

        TestStreamFunction fn = new TestStreamFunction();
        InputSplit is = new StreamInputSplit(uris, fn, new Random(12345));
        rr.initialize(is);

        List<List<List<Writable>>> act = new ArrayList<>();
        while (rr.hasNext()) {
            act.add(rr.sequenceRecord());
        }

        rr.reset();
        List<List<List<Writable>>> act2 = new ArrayList<>();
        while (rr.hasNext()) {
            act2.add(rr.sequenceRecord());
        }

        rr.reset();
        List<List<List<Writable>>> act3 = new ArrayList<>();
        while (rr.hasNext()) {
            act3.add(rr.sequenceRecord());
        }

        assertEquals(3, act.size());
        assertEquals(3, act2.size());
        assertEquals(3, act3.size());

        /*
        System.out.println(act);
        System.out.println("---------");
        System.out.println(act2);
        System.out.println("---------");
        System.out.println(act3);
        */

        //Check not the same. With this RNG seed, results are different for first 3 resets
        assertNotEquals(act, act2);
        assertNotEquals(act2, act3);
        assertNotEquals(act, act3);
    }


    public static class TestStreamFunction implements Function<URI,InputStream> {
        public List<URI> calledWithUris = new ArrayList<>();
        @Override
        public InputStream apply(URI uri) {
            calledWithUris.add(uri);        //Just for testing to ensure function is used
            try {
                return new FileInputStream(new File(uri));
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }
    }
}
