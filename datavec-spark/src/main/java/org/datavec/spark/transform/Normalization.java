package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrame;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;


import java.util.List;


/**
 * SImple dataframe based normalization.
 * Column based transforms such as mmin/max scaling
 * based on column min max and zero mean unit variance
 * using column wise statistics.
 *
 * @author Adam Gibson
 */
public class Normalization {



    /**
     * Normalize by zero mean unit variance
     * @param frame the data to normalize
     * @return a zero mean unit variance centered
     * rdd
     */
    public static DataFrame zeromeanUnitVariance(DataFrame frame) {
        String[] columnNames = frame.columns();
        for(String columnName : columnNames) {
            Column mean = DataFrames.mean(frame,columnName);
            Column std = DataFrames.std(frame,columnName);
            frame = frame.withColumn(columnName,frame.col(columnName).minus(mean).divide(std));
        }

        return frame;
    }

    /**
     * Normalize by zero mean unit variance
     * @param schema the schema to use
     *               to create the data frame
     * @param data the data to normalize
     * @return a zero mean unit variance centered
     * rdd
     */
    public static JavaRDD<List<Writable>> zeromeanUnitVariance(Schema schema, JavaRDD<List<Writable>> data) {
        DataFrame frame = DataFrames.toDataFrame(schema,data);
        return DataFrames.toRecords(zeromeanUnitVariance(frame)).getSecond();
    }

    public static JavaRDD<List<List<Writable>>> zeroMeanUnitVarianceSequence(Schema schema, JavaRDD<List<List<Writable>>> sequence){
        DataFrame frame = DataFrames.sequenceToDataFrame(schema, sequence);
    }

    /**
     * Scale based on min,max
     * @param dataFrame the dataframe to scale
     * @param min the minimum value
     * @param max the maximum value
     * @return the normalized dataframe per column
     */
    public static DataFrame normalize(DataFrame dataFrame,double min,double max) {
        String[] columnNames = dataFrame.columns();
        for(String columnName : columnNames) {
            Column min2 = DataFrames.min(dataFrame,columnName);
            Column max2 = DataFrames.max(dataFrame,columnName);
            Column maxMinusMin = max2.minus(min2);
            dataFrame = dataFrame.withColumn(columnName,dataFrame.col(columnName).minus(min2).divide(maxMinusMin.plus(1e-6)).multiply(max - min).plus(min));
        }

        return dataFrame;
    }

    /**
     * Scale based on min,max
     * @param schema the schema of the data to scale
     * @param data the data to sclae
     * @param min the minimum value
     * @param max the maximum value
     * @return the normalized ata
     */
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data,double min,double max) {
        DataFrame frame = DataFrames.toDataFrame(schema,data);
        return DataFrames.toRecords(normalize(frame,min,max)).getSecond();
    }



    /**
     * Scale based on min,max
     * @param dataFrame the dataframe to scale
     * @return the normalized dataframe per column
     */
    public static DataFrame normalize(DataFrame dataFrame) {
        return normalize(dataFrame,0,1);
    }

    /**
     * Scale all data  0 to 1
     * @param schema the schema of the data to scale
     * @param data the data to scale
     * @return the normalized ata
     */
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data) {
        return normalize(schema,data,0,1);
    }
}
