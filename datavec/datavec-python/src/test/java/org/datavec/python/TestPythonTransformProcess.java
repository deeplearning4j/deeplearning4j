package org.datavec.python;

import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.util.Arrays;
import java.util.List;
import static org.junit.Assert.assertEquals;


public class TestPythonTransformProcess {

    @Test
    public void testStringConcat() throws Exception{
        Schema.Builder schemaBuilder = new Schema.Builder();
        schemaBuilder
                .addColumnString("col1")
                .addColumnString("col2");

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnString("col3");
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";
        PythonTransformProcess ptp = new PythonTransformProcess(pythonCode, initialSchema, finalSchema);

        List<Writable> inputs = Arrays.asList((Writable) new Text("Hello "), new Text("World!"));

        List<Writable> outputs = ptp.execute(inputs);
        assertEquals((outputs.get(0)).toString(), "Hello ");
        assertEquals((outputs.get(1)).toString(), "World!");
        assertEquals((outputs.get(2)).toString(), "Hello World!");

    }

    @Test
    public void testNDArray() throws Exception{
        long[] shape = new long[]{3, 2};
        INDArray arr1 = Nd4j.rand(shape);
        INDArray arr2 = Nd4j.rand(shape);

        INDArray expectedOutput = arr1.add(arr2);

        Schema.Builder schemaBuilder = new Schema.Builder();
        schemaBuilder
                .addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape);

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnNDArray("col3", shape);
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";
        PythonTransformProcess ptp = new PythonTransformProcess(pythonCode, initialSchema, finalSchema);

        List<Writable> inputs = Arrays.asList(
                (Writable) new NDArrayWritable(arr1),
                 new NDArrayWritable(arr2)
        );

        List<Writable> outputs = ptp.execute(inputs);
        assertEquals(arr1, ((NDArrayWritable)outputs.get(0)).get());
        assertEquals(arr2, ((NDArrayWritable)outputs.get(1)).get());
        assertEquals(expectedOutput,((NDArrayWritable)outputs.get(2)).get());

    }
}
