package org.canova.api.records.reader.impl;

import org.canova.api.split.InputSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.junit.Test;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.Iterator;

import static org.junit.Assert.assertEquals;

public class CSVSequenceRecordReaderTest {

    @Test
    public void test() throws Exception {

        CSVSequenceRecordReader seqReader = new CSVSequenceRecordReader(1,",");
        seqReader.initialize(new TestInputSplit());

        int sequenceCount = 0;
        while(seqReader.hasNext()){
            Collection<Collection<Writable>> sequence = seqReader.sequenceRecord();
            assertEquals(4,sequence.size());    //4 lines, plus 1 header line

            Iterator<Collection<Writable>> timeStepIter = sequence.iterator();
            int lineCount = 0;
            while(timeStepIter.hasNext()){
                Collection<Writable> timeStep = timeStepIter.next();
                assertEquals(3,timeStep.size());
                Iterator<Writable> lineIter = timeStep.iterator();
                int countInLine = 0;
                while(lineIter.hasNext()){
                    Writable entry = lineIter.next();
                    int expValue = 100*sequenceCount + 10*lineCount + countInLine;
                    assertEquals(String.valueOf(expValue),entry.toString());
                    countInLine++;
                }
                lineCount++;
            }
            sequenceCount++;
        }
    }

    @Test
    public void testReset() throws Exception {
        CSVSequenceRecordReader seqReader = new CSVSequenceRecordReader(1,",");
        seqReader.initialize(new TestInputSplit());

        int nTests = 5;
        for( int i=0; i<nTests; i++ ) {
            seqReader.reset();

            int sequenceCount = 0;
            while (seqReader.hasNext()) {
                Collection<Collection<Writable>> sequence = seqReader.sequenceRecord();
                assertEquals(4, sequence.size());    //4 lines, plus 1 header line

                Iterator<Collection<Writable>> timeStepIter = sequence.iterator();
                int lineCount = 0;
                while (timeStepIter.hasNext()) {
                    timeStepIter.next();
                    lineCount++;
                }
                sequenceCount++;
                assertEquals(4,lineCount);
            }
            assertEquals(3,sequenceCount);
        }
    }

    private static class TestInputSplit implements InputSplit {

        @Override
        public long length() {
            return 3;
        }

        @Override
        public URI[] locations() {
            URI[] arr = new URI[3];
            try {
                arr[0] = new ClassPathResource("csvsequence_0.txt").getFile().toURI();
                arr[1] = new ClassPathResource("csvsequence_1.txt").getFile().toURI();
                arr[2] = new ClassPathResource("csvsequence_2.txt").getFile().toURI();
            } catch(Exception e ){
                throw new RuntimeException(e);
            }
            return arr;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public double toDouble(){
            throw new UnsupportedOperationException();
        }

        @Override
        public float toFloat(){
            throw new UnsupportedOperationException();
        }

        @Override
        public int toInt(){
            throw new UnsupportedOperationException();
        }

        @Override
        public long toLong(){
            throw new UnsupportedOperationException();
        }
    }
}
