package org.datavec.python;

import org.datavec.api.writable.Writable;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.junit.Test;

import java.util.ArrayList;
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

        List<Writable> inputs = Arrays.asList((Writable) new Text("Hello "), (Writable) new Text("World!"));

        List<Writable> outputs = ptp.execute(inputs);
        assertEquals(((Text)outputs.get(0)).toString(), "Hello ");
        assertEquals(((Text)outputs.get(1)).toString(), "World!");
        assertEquals(((Text)outputs.get(2)).toString(), "Hello World!");

    }
}
