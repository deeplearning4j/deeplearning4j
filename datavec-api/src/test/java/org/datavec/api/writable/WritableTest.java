package org.datavec.api.writable;

import org.datavec.api.writable.comparator.TextWritableComparator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class WritableTest {

    @Test
    public void testWritableEqualityReflexive() {
        assertEquals(new IntWritable(1), new IntWritable(1));
        assertEquals(new LongWritable(1), new LongWritable(1));
        assertEquals(new DoubleWritable(1), new DoubleWritable(1));
        assertEquals(new FloatWritable(1), new FloatWritable(1));
        assertEquals(new Text("Hello"), new Text("Hello"));

        INDArray ndArray = Nd4j.rand(new int[]{1, 100});

        assertEquals(new NDArrayWritable(ndArray), new NDArrayWritable(ndArray));
        assertEquals(new NullWritable(), new NullWritable());
        assertEquals(new BooleanWritable(true), new BooleanWritable(true));
        byte b = 0;
        assertEquals(new ByteWritable(b), new ByteWritable(b));
    }

    @Test
    public void testByteWritable() {
        byte b = 0xfffffffe;
        assertEquals(new IntWritable(-2), new ByteWritable(b));
        assertEquals(new LongWritable(-2), new ByteWritable(b));
        assertEquals(new ByteWritable(b), new IntWritable(-2));
        assertEquals(new ByteWritable(b), new LongWritable(-2));

        // those would cast to the same Int
        byte minus126 = 0xffffff82;
        assertNotEquals(new ByteWritable(minus126), new IntWritable(130));
    }

    @Test
    public void testIntLongWritable() {
        assertEquals(new IntWritable(1), new LongWritable(1l));
        assertEquals(new LongWritable(2l), new IntWritable(2));

        long l = 1L << 34;
        // those would cast to the same Int
        assertNotEquals(new LongWritable(l), new IntWritable(4));
    }


    @Test
    public void testDoubleFloatWritable(){
        assertEquals(new DoubleWritable(1d), new FloatWritable(1f));
        assertEquals(new FloatWritable(2f), new DoubleWritable(2d));

        // we defer to Java equality for Floats
        assertNotEquals(new DoubleWritable(1.1d), new FloatWritable(1.1f));
        // same idea as above
        assertNotEquals(new DoubleWritable(1.1d), new FloatWritable((float)1.1d));

        assertNotEquals(new DoubleWritable((double)Float.MAX_VALUE + 1), new FloatWritable(Float.POSITIVE_INFINITY));
    }


    @Test
    public void testFuzzies(){
        assertTrue(new DoubleWritable(1.1d).fuzzyEquals(new FloatWritable(1.1f), 1e-6d));
        assertTrue(new FloatWritable(1.1f).fuzzyEquals(new DoubleWritable(1.1d), 1e-6d));
        byte b = 0xfffffffe;
        assertTrue(new ByteWritable(b).fuzzyEquals(new DoubleWritable(-2.0), 1e-6d));
        assertFalse(new IntWritable(1).fuzzyEquals(new FloatWritable(1.1f), 1e-2d));
        assertTrue(new IntWritable(1).fuzzyEquals(new FloatWritable(1.05f), 1e-1d));
        assertTrue(new LongWritable(1).fuzzyEquals(new DoubleWritable(1.05f), 1e-1d));

    }
}
