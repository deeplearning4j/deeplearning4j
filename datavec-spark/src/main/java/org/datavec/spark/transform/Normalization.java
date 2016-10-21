package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.DataFrame;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;


import java.util.List;


/**
 * Created by agibsonccc on 10/21/16.
 */
public class Normalization {


    public static JavaRDD<List<Writable>> zeromeanUnitVariance(Schema schema, JavaRDD<List<Writable>> data) {
        DataFrame frame = DataFrames.toDataFrame(schema,data);
        String[] columnNames = frame.columns();
        DataFrame stats = frame.describe(columnNames);

        return DataFrames.toRecords(frame).getSecond();
    }



    public static JavaRDD<List<Writable>> max(Schema schema, JavaRDD<List<Writable>> data) {
        DataFrame frame = DataFrames.toDataFrame(schema,data);

        return DataFrames.toRecords(frame).getSecond();
    }


    public static JavaRDD<List<Writable>> min(Schema schema, JavaRDD<List<Writable>> data) {
        DataFrame frame = DataFrames.toDataFrame(schema,data);

        return DataFrames.toRecords(frame).getSecond();
    }


    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data) {
        DataFrame frame = DataFrames.toDataFrame(schema,data);

        return DataFrames.toRecords(frame).getSecond();
    }




}
