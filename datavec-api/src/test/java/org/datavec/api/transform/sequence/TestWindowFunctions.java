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

package org.datavec.api.transform.sequence;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.sequence.window.OverlappingTimeWindowFunction;
import org.datavec.api.transform.sequence.window.TimeWindowFunction;
import org.datavec.api.transform.sequence.window.WindowFunction;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 16/04/2016.
 */
public class TestWindowFunctions {

    @Test
    public void testTimeWindowFunction() {

        //Time windowing: 1 second (1000 milliseconds) window

        //Create some data.
        List<List<Writable>> sequence = new ArrayList<>();
        //First window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L), new IntWritable(0)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 100L), new IntWritable(1)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 200L), new IntWritable(2)));
        //Second window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 1000L), new IntWritable(3)));
        //Third window: empty
        //Fourth window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 3000L), new IntWritable(4)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 3100L), new IntWritable(5)));

        Schema schema = new SequenceSchema.Builder().addColumnTime("timecolumn", DateTimeZone.UTC)
                        .addColumnInteger("intcolumn").build();

        WindowFunction wf = new TimeWindowFunction("timecolumn", 1, TimeUnit.SECONDS);
        wf.setInputSchema(schema);

        List<List<List<Writable>>> windows = wf.applyToSequence(sequence);

        assertEquals(4, windows.size());
        assertEquals(3, windows.get(0).size());
        assertEquals(1, windows.get(1).size());
        assertEquals(0, windows.get(2).size());
        assertEquals(2, windows.get(3).size());

        List<List<Writable>> exp0 = new ArrayList<>();
        exp0.add(sequence.get(0));
        exp0.add(sequence.get(1));
        exp0.add(sequence.get(2));
        assertEquals(exp0, windows.get(0));

        List<List<Writable>> exp1 = new ArrayList<>();
        exp1.add(sequence.get(3));
        assertEquals(exp1, windows.get(1));

        List<List<Writable>> exp2 = new ArrayList<>();
        assertEquals(exp2, windows.get(2));

        List<List<Writable>> exp3 = new ArrayList<>();
        exp3.add(sequence.get(4));
        exp3.add(sequence.get(5));
        assertEquals(exp3, windows.get(3));
    }

    @Test
    public void testTimeWindowFunctionExcludeEmpty() {

        //Time windowing: 1 second (1000 milliseconds) window

        //Create some data.
        List<List<Writable>> sequence = new ArrayList<>();
        //First window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L), new IntWritable(0)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 100L), new IntWritable(1)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 200L), new IntWritable(2)));
        //Second window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 1000L), new IntWritable(3)));
        //Third window: empty
        //Fourth window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 3000L), new IntWritable(4)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 3100L), new IntWritable(5)));

        Schema schema = new SequenceSchema.Builder().addColumnTime("timecolumn", DateTimeZone.UTC)
                        .addColumnInteger("intcolumn").build();

        WindowFunction wf = new TimeWindowFunction.Builder().timeColumn("timecolumn").windowSize(1, TimeUnit.SECONDS)
                        .excludeEmptyWindows(true).build();

        wf.setInputSchema(schema);

        List<List<List<Writable>>> windows = wf.applyToSequence(sequence);

        assertEquals(3, windows.size());
        assertEquals(3, windows.get(0).size());
        assertEquals(1, windows.get(1).size());
        assertEquals(2, windows.get(2).size());

        List<List<Writable>> exp0 = new ArrayList<>();
        exp0.add(sequence.get(0));
        exp0.add(sequence.get(1));
        exp0.add(sequence.get(2));
        assertEquals(exp0, windows.get(0));

        List<List<Writable>> exp1 = new ArrayList<>();
        exp1.add(sequence.get(3));
        assertEquals(exp1, windows.get(1));

        List<List<Writable>> exp2 = new ArrayList<>();
        exp2.add(sequence.get(4));
        exp2.add(sequence.get(5));
        assertEquals(exp2, windows.get(2));
    }

    @Test
    public void testOverlappingTimeWindowFunctionSimple() {
        //Compare Overlapping and standard window functions where the window separation is equal to the window size
        // In this case, we should get exactly the same results from both.
        //Time windowing: 1 second (1000 milliseconds) window

        //Create some data.
        List<List<Writable>> sequence = new ArrayList<>();
        //First window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L), new IntWritable(0)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 100L), new IntWritable(1)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 200L), new IntWritable(2)));
        //Second window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 1000L), new IntWritable(3)));
        //Third window: empty
        //Fourth window:
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 3000L), new IntWritable(4)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1451606400000L + 3100L), new IntWritable(5)));

        Schema schema = new SequenceSchema.Builder().addColumnTime("timecolumn", DateTimeZone.UTC)
                        .addColumnInteger("intcolumn").build();

        WindowFunction wf = new TimeWindowFunction("timecolumn", 1, TimeUnit.SECONDS);
        wf.setInputSchema(schema);

        WindowFunction wf2 = new OverlappingTimeWindowFunction("timecolumn", 1, TimeUnit.SECONDS, 1, TimeUnit.SECONDS);
        wf2.setInputSchema(schema);

        List<List<List<Writable>>> windowsExp = wf.applyToSequence(sequence);
        List<List<List<Writable>>> windowsAct = wf2.applyToSequence(sequence);

        int[] expSizes = {3, 1, 0, 2};
        assertEquals(4, windowsExp.size());
        assertEquals(4, windowsAct.size());
        for (int i = 0; i < 4; i++) {
            assertEquals(expSizes[i], windowsExp.get(i).size());
            assertEquals(expSizes[i], windowsAct.get(i).size());

            assertEquals(windowsExp.get(i), windowsAct.get(i));
        }
    }

    @Test
    public void testOverlappingTimeWindowFunction() {
        //Create some data.
        List<List<Writable>> sequence = new ArrayList<>();
        //First window:
        sequence.add(Arrays.asList((Writable) new LongWritable(0), new IntWritable(0)));
        sequence.add(Arrays.asList((Writable) new LongWritable(100), new IntWritable(1)));
        sequence.add(Arrays.asList((Writable) new LongWritable(200), new IntWritable(2)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1000), new IntWritable(3)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1500), new IntWritable(4)));
        sequence.add(Arrays.asList((Writable) new LongWritable(2000), new IntWritable(5)));
        sequence.add(Arrays.asList((Writable) new LongWritable(5000), new IntWritable(7)));


        Schema schema = new SequenceSchema.Builder().addColumnTime("timecolumn", DateTimeZone.UTC)
                        .addColumnInteger("intcolumn").build();
        //Window size: 2 seconds; calculated every 1 second
        WindowFunction wf2 = new OverlappingTimeWindowFunction("timecolumn", 2, TimeUnit.SECONDS, 1, TimeUnit.SECONDS);
        wf2.setInputSchema(schema);

        List<List<List<Writable>>> windowsAct = wf2.applyToSequence(sequence);

        //First window: -1000 to 1000
        List<List<Writable>> exp0 = new ArrayList<>();
        exp0.add(Arrays.asList((Writable) new LongWritable(0), new IntWritable(0)));
        exp0.add(Arrays.asList((Writable) new LongWritable(100), new IntWritable(1)));
        exp0.add(Arrays.asList((Writable) new LongWritable(200), new IntWritable(2)));
        //Second window: 0 to 2000
        List<List<Writable>> exp1 = new ArrayList<>();
        exp1.add(Arrays.asList((Writable) new LongWritable(0), new IntWritable(0)));
        exp1.add(Arrays.asList((Writable) new LongWritable(100), new IntWritable(1)));
        exp1.add(Arrays.asList((Writable) new LongWritable(200), new IntWritable(2)));
        exp1.add(Arrays.asList((Writable) new LongWritable(1000), new IntWritable(3)));
        exp1.add(Arrays.asList((Writable) new LongWritable(1500), new IntWritable(4)));
        //Third window: 1000 to 3000
        List<List<Writable>> exp2 = new ArrayList<>();
        exp2.add(Arrays.asList((Writable) new LongWritable(1000), new IntWritable(3)));
        exp2.add(Arrays.asList((Writable) new LongWritable(1500), new IntWritable(4)));
        exp2.add(Arrays.asList((Writable) new LongWritable(2000), new IntWritable(5)));
        //Fourth window: 2000 to 4000
        List<List<Writable>> exp3 = new ArrayList<>();
        exp3.add(Arrays.asList((Writable) new LongWritable(2000), new IntWritable(5)));
        //Fifth window: 3000 to 5000
        List<List<Writable>> exp4 = new ArrayList<>();
        //Sixth window: 4000 to 6000
        List<List<Writable>> exp5 = new ArrayList<>();
        exp5.add(Arrays.asList((Writable) new LongWritable(5000), new IntWritable(7)));
        //Seventh window: 5000 to 7000
        List<List<Writable>> exp6 = new ArrayList<>();
        exp6.add(Arrays.asList((Writable) new LongWritable(5000), new IntWritable(7)));

        List<List<List<Writable>>> windowsExp = Arrays.asList(exp0, exp1, exp2, exp3, exp4, exp5, exp6);

        assertEquals(7, windowsAct.size());
        for (int i = 0; i < 7; i++) {
            List<List<Writable>> exp = windowsExp.get(i);
            List<List<Writable>> act = windowsAct.get(i);

            assertEquals(exp, act);
        }
    }

    @Test
    public void testOverlappingTimeWindowFunctionExcludeEmpty() {
        //Create some data.
        List<List<Writable>> sequence = new ArrayList<>();
        //First window:
        sequence.add(Arrays.asList((Writable) new LongWritable(0), new IntWritable(0)));
        sequence.add(Arrays.asList((Writable) new LongWritable(100), new IntWritable(1)));
        sequence.add(Arrays.asList((Writable) new LongWritable(200), new IntWritable(2)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1000), new IntWritable(3)));
        sequence.add(Arrays.asList((Writable) new LongWritable(1500), new IntWritable(4)));
        sequence.add(Arrays.asList((Writable) new LongWritable(2000), new IntWritable(5)));
        sequence.add(Arrays.asList((Writable) new LongWritable(5000), new IntWritable(7)));


        Schema schema = new SequenceSchema.Builder().addColumnTime("timecolumn", DateTimeZone.UTC)
                        .addColumnInteger("intcolumn").build();
        //Window size: 2 seconds; calculated every 1 second
        //        WindowFunction wf2 = new OverlappingTimeWindowFunction("timecolumn",2,TimeUnit.SECONDS,1,TimeUnit.SECONDS);
        WindowFunction wf2 = new OverlappingTimeWindowFunction.Builder().timeColumn("timecolumn")
                        .windowSize(2, TimeUnit.SECONDS).windowSeparation(1, TimeUnit.SECONDS).excludeEmptyWindows(true)
                        .build();
        wf2.setInputSchema(schema);

        List<List<List<Writable>>> windowsAct = wf2.applyToSequence(sequence);

        //First window: -1000 to 1000
        List<List<Writable>> exp0 = new ArrayList<>();
        exp0.add(Arrays.asList((Writable) new LongWritable(0), new IntWritable(0)));
        exp0.add(Arrays.asList((Writable) new LongWritable(100), new IntWritable(1)));
        exp0.add(Arrays.asList((Writable) new LongWritable(200), new IntWritable(2)));
        //Second window: 0 to 2000
        List<List<Writable>> exp1 = new ArrayList<>();
        exp1.add(Arrays.asList((Writable) new LongWritable(0), new IntWritable(0)));
        exp1.add(Arrays.asList((Writable) new LongWritable(100), new IntWritable(1)));
        exp1.add(Arrays.asList((Writable) new LongWritable(200), new IntWritable(2)));
        exp1.add(Arrays.asList((Writable) new LongWritable(1000), new IntWritable(3)));
        exp1.add(Arrays.asList((Writable) new LongWritable(1500), new IntWritable(4)));
        //Third window: 1000 to 3000
        List<List<Writable>> exp2 = new ArrayList<>();
        exp2.add(Arrays.asList((Writable) new LongWritable(1000), new IntWritable(3)));
        exp2.add(Arrays.asList((Writable) new LongWritable(1500), new IntWritable(4)));
        exp2.add(Arrays.asList((Writable) new LongWritable(2000), new IntWritable(5)));
        //Fourth window: 2000 to 4000
        List<List<Writable>> exp3 = new ArrayList<>();
        exp3.add(Arrays.asList((Writable) new LongWritable(2000), new IntWritable(5)));
        //Fifth window: 3000 to 5000 -> Empty: excluded
        //Sixth window: 4000 to 6000
        List<List<Writable>> exp5 = new ArrayList<>();
        exp5.add(Arrays.asList((Writable) new LongWritable(5000), new IntWritable(7)));
        //Seventh window: 5000 to 7000
        List<List<Writable>> exp6 = new ArrayList<>();
        exp6.add(Arrays.asList((Writable) new LongWritable(5000), new IntWritable(7)));

        List<List<List<Writable>>> windowsExp = Arrays.asList(exp0, exp1, exp2, exp3, exp5, exp6);

        assertEquals(6, windowsAct.size());
        for (int i = 0; i < 6; i++) {
            List<List<Writable>> exp = windowsExp.get(i);
            List<List<Writable>> act = windowsAct.get(i);

            assertEquals(exp, act);
        }
    }

}
