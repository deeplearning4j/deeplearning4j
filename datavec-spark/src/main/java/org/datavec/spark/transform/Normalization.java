package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.*;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


/**
 * Simple dataframe based normalization.
 * Column based transforms such as min/max scaling
 * based on column min max and zero mean unit variance
 * using column wise statistics.
 *
 * @author Adam Gibson
 */
public class Normalization {


    /**
     * Normalize by zero mean unit variance
     *
     * @param frame the data to normalize
     * @return a zero mean unit variance centered
     * rdd
     */
    public static DataFrame zeromeanUnitVariance(DataFrame frame) {
        return zeromeanUnitVariance(frame, Collections.<String>emptyList());
    }

    /**
     * Normalize by zero mean unit variance
     *
     * @param schema the schema to use
     *               to create the data frame
     * @param data   the data to normalize
     * @return a zero mean unit variance centered
     * rdd
     */
    public static JavaRDD<List<Writable>> zeromeanUnitVariance(Schema schema, JavaRDD<List<Writable>> data) {
        return zeromeanUnitVariance(schema, data, Collections.<String>emptyList());
    }

    /**
     * Scale based on min,max
     *
     * @param dataFrame the dataframe to scale
     * @param min       the minimum value
     * @param max       the maximum value
     * @return the normalized dataframe per column
     */
    public static DataFrame normalize(DataFrame dataFrame, double min, double max) {
        return normalize(dataFrame, min, max, Collections.<String>emptyList());
    }

    /**
     * Scale based on min,max
     *
     * @param schema the schema of the data to scale
     * @param data   the data to sclae
     * @param min    the minimum value
     * @param max    the maximum value
     * @return the normalized ata
     */
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data, double min, double max) {
        DataFrame frame = DataFrames.toDataFrame(schema, data);
        return DataFrames.toRecords(normalize(frame, min, max, Collections.<String>emptyList())).getSecond();
    }


    /**
     * Scale based on min,max
     *
     * @param dataFrame the dataframe to scale
     * @return the normalized dataframe per column
     */
    public static DataFrame normalize(DataFrame dataFrame) {
        return normalize(dataFrame, 0, 1, Collections.<String>emptyList());
    }

    /**
     * Scale all data  0 to 1
     *
     * @param schema the schema of the data to scale
     * @param data   the data to scale
     * @return the normalized ata
     */
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data) {
        return normalize(schema, data, 0, 1, Collections.<String>emptyList());
    }


    /**
     * Normalize by zero mean unit variance
     *
     * @param frame the data to normalize
     * @return a zero mean unit variance centered
     * rdd
     */
    public static DataFrame zeromeanUnitVariance(DataFrame frame, List<String> skipColumns) {
        String[] columnNames = frame.columns();
        for (String columnName : columnNames) {
            if (skipColumns.contains(columnName)) continue;

            DataFrame meanStd = frame.select(columnName).agg(mean(columnName), stddev(columnName));
            Row r = meanStd.collect()[0];
            double mean = ((Number) r.get(0)).doubleValue();
            double std = ((Number) r.get(1)).doubleValue();
            if (std == 0.0) std = 1; //All same value -> (x-x)/1 = 0

            frame = frame.withColumn(columnName, frame.col(columnName).minus(mean).divide(std));
        }

        return frame;
    }

    /**
     * Normalize by zero mean unit variance
     *
     * @param schema the schema to use
     *               to create the data frame
     * @param data   the data to normalize
     * @return a zero mean unit variance centered
     * rdd
     */
    public static JavaRDD<List<Writable>> zeromeanUnitVariance(Schema schema, JavaRDD<List<Writable>> data, List<String> skipColumns) {
        DataFrame frame = DataFrames.toDataFrame(schema, data);
        return DataFrames.toRecords(zeromeanUnitVariance(frame, skipColumns)).getSecond();
    }

    /**
     * Normalize the sequence by zero mean unit variance
     *
     * @param schema   Schema of the data to normalize
     * @param sequence Sequence data
     * @return Normalized sequence
     */
    public static JavaRDD<List<List<Writable>>> zeroMeanUnitVarianceSequence(Schema schema, JavaRDD<List<List<Writable>>> sequence) {
        return zeroMeanUnitVarianceSequence(schema, sequence, null);
    }

    /**
     * Normalize the sequence by zero mean unit variance
     *
     * @param schema         Schema of the data to normalize
     * @param sequence       Sequence data
     * @param excludeColumns List of  columns to exclude from the normalization
     * @return Normalized sequence
     */
    public static JavaRDD<List<List<Writable>>> zeroMeanUnitVarianceSequence(Schema schema, JavaRDD<List<List<Writable>>> sequence,
                                                                             List<String> excludeColumns) {
        DataFrame frame = DataFrames.toDataFrameSequence(schema, sequence);
        if (excludeColumns == null)
            excludeColumns = Arrays.asList(DataFrames.SEQUENCE_UUID_COLUMN, DataFrames.SEQUENCE_INDEX_COLUMN);
        else {
            excludeColumns = new ArrayList<>(excludeColumns);
            excludeColumns.add(DataFrames.SEQUENCE_UUID_COLUMN);
            excludeColumns.add(DataFrames.SEQUENCE_INDEX_COLUMN);
        }
        frame = zeromeanUnitVariance(frame, excludeColumns);
        return DataFrames.toRecordsSequence(frame).getSecond();
    }

    /**
     * Scale based on min,max
     *
     * @param dataFrame the dataframe to scale
     * @param min       the minimum value
     * @param max       the maximum value
     * @return the normalized dataframe per column
     */
    public static DataFrame normalize(DataFrame dataFrame, double min, double max, List<String> skipColumns) {
        String[] columnNames = dataFrame.columns();

        for (String columnName : columnNames) {
            if (skipColumns.contains(columnName))
                continue;
            DataFrame minMax = dataFrame.select(columnName).agg(min(columnName), max(columnName));
            Row r = minMax.collect()[0];
            double dMin = ((Number) r.get(0)).doubleValue();
            double dMax = ((Number) r.get(1)).doubleValue();

            double maxSubMin = dMax - dMin;
            if (maxSubMin == 0) maxSubMin = 1;

            Column newCol = dataFrame.col(columnName).minus(dMin).divide(maxSubMin).multiply(max - min).plus(min);
            dataFrame = dataFrame.withColumn(columnName, newCol);
        }

        return dataFrame;
    }

    /**
     * Scale based on min,max
     *
     * @param schema the schema of the data to scale
     * @param data   the data to sclae
     * @param min    the minimum value
     * @param max    the maximum value
     * @return the normalized ata
     */
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data, double min, double max, List<String> skipColumns) {
        DataFrame frame = DataFrames.toDataFrame(schema, data);
        return DataFrames.toRecords(normalize(frame, min, max, skipColumns)).getSecond();
    }

    public static JavaRDD<List<List<Writable>>> normalizeSequence(Schema schema, JavaRDD<List<List<Writable>>> data) {
        return normalizeSequence(schema, data, 0, 1);
    }

    /**
     * Normalize each column of a sequence, based on min/max
     *
     * @param schema Schema of the data
     * @param data   Data to normalize
     * @param min    New minimum value
     * @param max    New maximum value
     * @return Normalized data
     */
    public static JavaRDD<List<List<Writable>>> normalizeSequence(Schema schema, JavaRDD<List<List<Writable>>> data, double min, double max) {
        return normalizeSequence(schema, data, min, max, null);
    }

    /**
     * Normalize each column of a sequence, based on min/max
     *
     * @param schema         Schema of the data
     * @param data           Data to normalize
     * @param min            New minimum value
     * @param max            New maximum value
     * @param excludeColumns List of columns to exclude
     * @return Normalized data
     */
    public static JavaRDD<List<List<Writable>>> normalizeSequence(Schema schema, JavaRDD<List<List<Writable>>> data,
                                                                  double min, double max, List<String> excludeColumns) {
        if (excludeColumns == null)
            excludeColumns = Arrays.asList(DataFrames.SEQUENCE_UUID_COLUMN, DataFrames.SEQUENCE_INDEX_COLUMN);
        else {
            excludeColumns = new ArrayList<>(excludeColumns);
            excludeColumns.add(DataFrames.SEQUENCE_UUID_COLUMN);
            excludeColumns.add(DataFrames.SEQUENCE_INDEX_COLUMN);
        }
        DataFrame frame = DataFrames.toDataFrameSequence(schema, data);
        return DataFrames.toRecordsSequence(normalize(frame, min, max, excludeColumns)).getSecond();
    }


    /**
     * Scale based on min,max
     *
     * @param dataFrame the dataframe to scale
     * @return the normalized dataframe per column
     */
    public static DataFrame normalize(DataFrame dataFrame, List<String> skipColumns) {
        return normalize(dataFrame, 0, 1, skipColumns);
    }

    /**
     * Scale all data  0 to 1
     *
     * @param schema the schema of the data to scale
     * @param data   the data to scale
     * @return the normalized ata
     */
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data, List<String> skipColumns) {
        return normalize(schema, data, 0, 1, skipColumns);
    }
}
