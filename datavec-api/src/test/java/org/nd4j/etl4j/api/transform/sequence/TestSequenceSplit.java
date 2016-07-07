package org.nd4j.etl4j.api.transform.sequence;

import org.nd4j.etl4j.api.transform.schema.Schema;
import org.nd4j.etl4j.api.transform.schema.SequenceSchema;
import org.nd4j.etl4j.api.transform.sequence.split.SequenceSplitTimeSeparation;
import org.nd4j.etl4j.api.io.data.LongWritable;
import org.nd4j.etl4j.api.io.data.Text;
import org.nd4j.etl4j.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 19/04/2016.
 */
public class TestSequenceSplit {

    @Test
    public void testSequenceSplitTimeSeparation(){

        Schema schema = new SequenceSchema.Builder()
                .addColumnTime("time", DateTimeZone.UTC)
                .addColumnString("text")
                .build();

        List<List<Writable>> inputSequence = new ArrayList<>();
        inputSequence.add(Arrays.asList((Writable)new LongWritable(0), new Text("t0")));
        inputSequence.add(Arrays.asList((Writable)new LongWritable(1000), new Text("t1")));
            //Second split: 74 seconds later
        inputSequence.add(Arrays.asList((Writable)new LongWritable(75000), new Text("t2")));
        inputSequence.add(Arrays.asList((Writable)new LongWritable(100000), new Text("t3")));
            //Third split: 1 minute and 1 milliseconds later
        inputSequence.add(Arrays.asList((Writable)new LongWritable(160001), new Text("t4")));

        SequenceSplit seqSplit = new SequenceSplitTimeSeparation("time",1, TimeUnit.MINUTES);
        seqSplit.setInputSchema(schema);

        List<List<List<Writable>>> splits = seqSplit.split(inputSequence);
        assertEquals(3, splits.size());

        List<List<Writable>> exp0 = new ArrayList<>();
        exp0.add(Arrays.asList((Writable)new LongWritable(0), new Text("t0")));
        exp0.add(Arrays.asList((Writable)new LongWritable(1000), new Text("t1")));
        List<List<Writable>> exp1 = new ArrayList<>();
        exp1.add(Arrays.asList((Writable)new LongWritable(75000), new Text("t2")));
        exp1.add(Arrays.asList((Writable)new LongWritable(100000), new Text("t3")));
        List<List<Writable>> exp2 = new ArrayList<>();
        exp2.add(Arrays.asList((Writable)new LongWritable(160001), new Text("t4")));

        assertEquals(exp0, splits.get(0));
        assertEquals(exp1, splits.get(1));
        assertEquals(exp2, splits.get(2));
    }

}
