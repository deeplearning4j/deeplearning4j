package org.datavec.arrow.recordreader;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.ArrowConverter;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class ArrowWritableRecordTimeSeriesBatchTests {

    private static BufferAllocator bufferAllocator = new RootAllocator(Long.MAX_VALUE);


    @Test
    public void testBasicIndexing() {
        Schema.Builder schema = new Schema.Builder();
        for(int i = 0; i < 3; i++) {
            schema.addColumnInteger(String.valueOf(i));
        }


        List<List<Writable>> timeStep = Arrays.asList(
                Arrays.<Writable>asList(new IntWritable(0),new IntWritable(1),new IntWritable(2)),
                Arrays.<Writable>asList(new IntWritable(0),new IntWritable(1),new IntWritable(2)),
                Arrays.<Writable>asList(new IntWritable(0),new IntWritable(1),new IntWritable(2))
        );

        int numTimeSteps = 5;
        List<List<List<Writable>>> timeSteps = new ArrayList<>(numTimeSteps);
        for(int i = 0; i < numTimeSteps; i++) {
            timeSteps.add(timeStep);
        }

        List<FieldVector> fieldVectors = ArrowConverter.toArrowColumnsTimeSeries(bufferAllocator, schema.build(), timeSteps);
        assertEquals(3,fieldVectors.size());
        ArrowWritableRecordTimeSeriesBatch arrowWritableRecordTimeSeriesBatch = new ArrowWritableRecordTimeSeriesBatch(fieldVectors,schema.build(),timeStep.size() * timeStep.get(0).size());

        List<List<List<Writable>>> timeStepsTest = new ArrayList<>(arrowWritableRecordTimeSeriesBatch);

        assertEquals(timeSteps,timeStepsTest);

    }

}
