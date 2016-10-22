package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.DataFrame;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/22/16.
 */
public class NormalizationTests extends BaseSparkTest {

    @Test
    public void normalizationTests() {
        List<List<Writable>> data = new ArrayList<>();
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 6;
        for(int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));

        for(int i = 0; i < 5; i++) {
            List<Writable> record = new ArrayList<>(numColumns);
            data.add(record);
            for(int j = 0; j < numColumns; j++) {
                record.add(new DoubleWritable(1.0));
            }

        }

        Schema schema = builder.build();
        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        assertEquals(schema,DataFrames.fromStructType(DataFrames.fromSchema(schema)));
        assertEquals(rdd.collect(),DataFrames.toRecords(DataFrames.toDataFrame(schema,rdd)).getSecond().collect());

        DataFrame dataFrame = DataFrames.toDataFrame(schema,rdd);
        dataFrame.show();
        Normalization.zeromeanUnitVariance(dataFrame).show();
        Normalization.normalize(dataFrame).show();

    }

}
