package org.datavec.spark.transform;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.model.CSVRecord;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 12/24/16.
 */
public class CSVSparkTransformerTest {
    @Test
    public void testTransformer() {
      List<Writable> input = new ArrayList<>();
      input.add(new DoubleWritable(1.0));
      input.add(new DoubleWritable(2.0));

        Schema schema = new Schema.Builder()
                .addColumnDouble("1.0").addColumnDouble("2.0").build();
        List<Writable> output = new ArrayList<>();
        output.add(new Text("1.0"));
        output.add(new Text("2.0"));

        TransformProcess transformProcess = new TransformProcess.Builder(schema).convertToString("1.0").convertToString("2.0").build();
        CSVSparkTransform csvSparkTransform = new CSVSparkTransform(transformProcess);
        String[] values = new String[] {"1.0","2.0"};
        CSVRecord record = csvSparkTransform.transform(new CSVRecord(values));


    }

}
