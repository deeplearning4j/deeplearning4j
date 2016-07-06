package org.canova.api.records.reader.impl;

import org.canova.api.io.data.IntWritable;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 21/05/2016.
 */
public class TestCollectionRecordReaders {

    @Test
    public void testCollectionSequenceRecordReader(){

        List<List<List<Writable>>> listOfSequences = new ArrayList<>();

        List<List<Writable>> sequence1 = new ArrayList<>();
        sequence1.add(Arrays.asList((Writable)new IntWritable(0), new IntWritable(1)));
        sequence1.add(Arrays.asList((Writable)new IntWritable(2), new IntWritable(3)));
        listOfSequences.add(sequence1);

        List<List<Writable>> sequence2 = new ArrayList<>();
        sequence2.add(Arrays.asList((Writable)new IntWritable(4), new IntWritable(5)));
        sequence2.add(Arrays.asList((Writable)new IntWritable(6), new IntWritable(7)));
        listOfSequences.add(sequence2);

        SequenceRecordReader seqRR = new CollectionSequenceRecordReader(listOfSequences);
        assertTrue(seqRR.hasNext());

        assertEquals(sequence1, seqRR.sequenceRecord());
        assertEquals(sequence2, seqRR.sequenceRecord());
        assertFalse(seqRR.hasNext());

        seqRR.reset();
        assertEquals(sequence1, seqRR.sequenceRecord());
        assertEquals(sequence2, seqRR.sequenceRecord());
        assertFalse(seqRR.hasNext());
    }

}
