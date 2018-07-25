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
