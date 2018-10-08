package org.datavec.api.records.reader.impl;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVMultiSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class CSVMultiSequenceRecordReaderTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testConcatMode() throws Exception {
        for( int i=0; i<3; i++ ) {

            String seqSep;
            String seqSepRegex;
            switch (i){
                case 0:
                    seqSep = "";
                    seqSepRegex = "^$";
                    break;
                case 1:
                    seqSep = "---";
                    seqSepRegex = seqSep;
                    break;
                case 2:
                    seqSep = "&";
                    seqSepRegex = seqSep;
                    break;
                default:
                    throw new RuntimeException();
            }

            String str = "a,b,c\n1,2,3,4\nx,y\n" + seqSep + "\nA,B,C";
            File f = testDir.newFile();
            FileUtils.writeStringToFile(f, str);

            SequenceRecordReader seqRR = new CSVMultiSequenceRecordReader(seqSepRegex, CSVMultiSequenceRecordReader.Mode.CONCAT);
            seqRR.initialize(new FileSplit(f));


            List<List<Writable>> exp0 = new ArrayList<>();
            for (String s : "a,b,c,1,2,3,4,x,y".split(",")) {
                exp0.add(Collections.<Writable>singletonList(new Text(s)));
            }

            List<List<Writable>> exp1 = new ArrayList<>();
            for (String s : "A,B,C".split(",")) {
                exp1.add(Collections.<Writable>singletonList(new Text(s)));
            }

            assertEquals(exp0, seqRR.sequenceRecord());
            assertEquals(exp1, seqRR.sequenceRecord());
            assertFalse(seqRR.hasNext());

            seqRR.reset();

            assertEquals(exp0, seqRR.sequenceRecord());
            assertEquals(exp1, seqRR.sequenceRecord());
            assertFalse(seqRR.hasNext());
        }
    }

    @Test
    public void testEqualLength() throws Exception {

        for( int i=0; i<3; i++ ) {

            String seqSep;
            String seqSepRegex;
            switch (i) {
                case 0:
                    seqSep = "";
                    seqSepRegex = "^$";
                    break;
                case 1:
                    seqSep = "---";
                    seqSepRegex = seqSep;
                    break;
                case 2:
                    seqSep = "&";
                    seqSepRegex = seqSep;
                    break;
                default:
                    throw new RuntimeException();
            }

            String str = "a,b\n1,2\nx,y\n" + seqSep + "\nA\nB\nC";
            File f = testDir.newFile();
            FileUtils.writeStringToFile(f, str);

            SequenceRecordReader seqRR = new CSVMultiSequenceRecordReader(seqSepRegex, CSVMultiSequenceRecordReader.Mode.EQUAL_LENGTH);
            seqRR.initialize(new FileSplit(f));


            List<List<Writable>> exp0 = Arrays.asList(
                    Arrays.<Writable>asList(new Text("a"), new Text("1"), new Text("x")),
                    Arrays.<Writable>asList(new Text("b"), new Text("2"), new Text("y")));

            List<List<Writable>> exp1 = Collections.singletonList(Arrays.<Writable>asList(new Text("A"), new Text("B"), new Text("C")));

            assertEquals(exp0, seqRR.sequenceRecord());
            assertEquals(exp1, seqRR.sequenceRecord());
            assertFalse(seqRR.hasNext());

            seqRR.reset();

            assertEquals(exp0, seqRR.sequenceRecord());
            assertEquals(exp1, seqRR.sequenceRecord());
            assertFalse(seqRR.hasNext());
        }
    }

    @Test
    public void testPadding() throws Exception {

        for( int i=0; i<3; i++ ) {

            String seqSep;
            String seqSepRegex;
            switch (i) {
                case 0:
                    seqSep = "";
                    seqSepRegex = "^$";
                    break;
                case 1:
                    seqSep = "---";
                    seqSepRegex = seqSep;
                    break;
                case 2:
                    seqSep = "&";
                    seqSepRegex = seqSep;
                    break;
                default:
                    throw new RuntimeException();
            }

            String str = "a,b\n1\nx\n" + seqSep + "\nA\nB\nC";
            File f = testDir.newFile();
            FileUtils.writeStringToFile(f, str);

            SequenceRecordReader seqRR = new CSVMultiSequenceRecordReader(seqSepRegex, CSVMultiSequenceRecordReader.Mode.PAD, new Text("PAD"));
            seqRR.initialize(new FileSplit(f));


            List<List<Writable>> exp0 = Arrays.asList(
                    Arrays.<Writable>asList(new Text("a"), new Text("1"), new Text("x")),
                    Arrays.<Writable>asList(new Text("b"), new Text("PAD"), new Text("PAD")));

            List<List<Writable>> exp1 = Collections.singletonList(Arrays.<Writable>asList(new Text("A"), new Text("B"), new Text("C")));

            assertEquals(exp0, seqRR.sequenceRecord());
            assertEquals(exp1, seqRR.sequenceRecord());
            assertFalse(seqRR.hasNext());

            seqRR.reset();

            assertEquals(exp0, seqRR.sequenceRecord());
            assertEquals(exp1, seqRR.sequenceRecord());
            assertFalse(seqRR.hasNext());
        }
    }
}
