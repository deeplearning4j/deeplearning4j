package org.datavec.arrow.recordreader;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.ArrowConverter;
import org.junit.Ignore;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

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
                Arrays.<Writable>asList(new IntWritable(1),new IntWritable(2),new IntWritable(3)),
                Arrays.<Writable>asList(new IntWritable(4),new IntWritable(5),new IntWritable(6))
        );

        int numTimeSteps = 5;
        List<List<List<Writable>>> timeSteps = new ArrayList<>(numTimeSteps);
        for(int i = 0; i < numTimeSteps; i++) {
            timeSteps.add(timeStep);
        }

        List<FieldVector> fieldVectors = ArrowConverter.toArrowColumnsTimeSeries(bufferAllocator, schema.build(), timeSteps);
        assertEquals(3,fieldVectors.size());
        for(FieldVector fieldVector : fieldVectors) {
            for(int i = 0; i < fieldVector.getValueCount(); i++) {
                assertFalse("Index " + i + " was null for field vector " + fieldVector, fieldVector.isNull(i));
            }
        }

        ArrowWritableRecordTimeSeriesBatch arrowWritableRecordTimeSeriesBatch = new ArrowWritableRecordTimeSeriesBatch(fieldVectors,schema.build(),timeStep.size() * timeStep.get(0).size());
        assertEquals(timeSteps,arrowWritableRecordTimeSeriesBatch.toArrayList());
    }

    @Test
    //not worried about this till after next release
    @Ignore
    public void testVariableLengthTS() {
        Schema.Builder schema = new Schema.Builder()
                .addColumnString("str")
                .addColumnInteger("int")
                .addColumnDouble("dbl");

        List<List<Writable>> firstSeq = Arrays.asList(
                Arrays.<Writable>asList(new Text("00"),new IntWritable(0),new DoubleWritable(2.0)),
                Arrays.<Writable>asList(new Text("01"),new IntWritable(1),new DoubleWritable(2.1)),
                Arrays.<Writable>asList(new Text("02"),new IntWritable(2),new DoubleWritable(2.2)));

        List<List<Writable>> secondSeq = Arrays.asList(
                Arrays.<Writable>asList(new Text("10"),new IntWritable(10),new DoubleWritable(12.0)),
                Arrays.<Writable>asList(new Text("11"),new IntWritable(11),new DoubleWritable(12.1)));

        List<List<List<Writable>>> sequences = Arrays.asList(firstSeq, secondSeq);


        List<FieldVector> fieldVectors = ArrowConverter.toArrowColumnsTimeSeries(bufferAllocator, schema.build(), sequences);
        assertEquals(3,fieldVectors.size());

        int timeSeriesStride = -1;  //Can't sequences of different length...
        ArrowWritableRecordTimeSeriesBatch arrowWritableRecordTimeSeriesBatch
                = new ArrowWritableRecordTimeSeriesBatch(fieldVectors,schema.build(),timeSeriesStride);

        List<List<List<Writable>>> asList = arrowWritableRecordTimeSeriesBatch.toArrayList();
        assertEquals(sequences, asList);
    }
  

}
