/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.spark.transform;

import org.apache.commons.collections.map.ListOrderedMap;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.util.*;


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
    public static Dataset<Row> zeromeanUnitVariance(Dataset<Row> frame) {
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
    public static Dataset<Row> normalize(Dataset<Row> dataFrame, double min, double max) {
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
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data, double min,
                    double max) {
        Dataset<Row> frame = DataFrames.toDataFrame(schema, data);
        return DataFrames.toRecords(normalize(frame, min, max, Collections.<String>emptyList())).getSecond();
    }


    /**
     * Scale based on min,max
     *
     * @param dataFrame the dataframe to scale
     * @return the normalized dataframe per column
     */
    public static Dataset<Row> normalize(Dataset<Row> dataFrame) {
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
    public static Dataset<Row> zeromeanUnitVariance(Dataset<Row> frame, List<String> skipColumns) {
        List<String> columnsList = DataFrames.toList(frame.columns());
        columnsList.removeAll(skipColumns);
        String[] columnNames = DataFrames.toArray(columnsList);
        //first row is std second row is mean, each column in a row is for a particular column
        List<Row> stdDevMean = stdDevMeanColumns(frame, columnNames);
        for (int i = 0; i < columnNames.length; i++) {
            String columnName = columnNames[i];
            double std = ((Number) stdDevMean.get(0).get(i)).doubleValue();
            double mean = ((Number) stdDevMean.get(1).get(i)).doubleValue();
            if (std == 0.0)
                std = 1; //All same value -> (x-x)/1 = 0

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
    public static JavaRDD<List<Writable>> zeromeanUnitVariance(Schema schema, JavaRDD<List<Writable>> data,
                    List<String> skipColumns) {
        Dataset<Row> frame = DataFrames.toDataFrame(schema, data);
        return DataFrames.toRecords(zeromeanUnitVariance(frame, skipColumns)).getSecond();
    }

    /**
     * Normalize the sequence by zero mean unit variance
     *
     * @param schema   Schema of the data to normalize
     * @param sequence Sequence data
     * @return Normalized sequence
     */
    public static JavaRDD<List<List<Writable>>> zeroMeanUnitVarianceSequence(Schema schema,
                    JavaRDD<List<List<Writable>>> sequence) {
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
    public static JavaRDD<List<List<Writable>>> zeroMeanUnitVarianceSequence(Schema schema,
                    JavaRDD<List<List<Writable>>> sequence, List<String> excludeColumns) {
        Dataset<Row> frame = DataFrames.toDataFrameSequence(schema, sequence);
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
     * Returns the min and max of the given columns
     * @param data the data to get the max for
     * @param columns the columns to get the
     * @return
     */
    public static List<Row> minMaxColumns(Dataset<Row> data, List<String> columns) {
        String[] arr = new String[columns.size()];
        for (int i = 0; i < arr.length; i++)
            arr[i] = columns.get(i);
        return minMaxColumns(data, arr);
    }

    /**
     * Returns the min and max of the given columns.
     * The list returned is a list of size 2 where each row
     * @param data the data to get the max for
     * @param columns the columns to get the
     * @return
     */
    public static List<Row> minMaxColumns(Dataset<Row> data, String... columns) {
        return aggregate(data, columns, new String[] {"min", "max"});
    }


    /**
     * Returns the standard deviation and mean of the given columns
     * @param data the data to get the max for
     * @param columns the columns to get the
     * @return
     */
    public static List<Row> stdDevMeanColumns(Dataset<Row> data, List<String> columns) {
        String[] arr = new String[columns.size()];
        for (int i = 0; i < arr.length; i++)
            arr[i] = columns.get(i);
        return stdDevMeanColumns(data, arr);
    }

    /**
     * Returns the standard deviation
     * and mean of the given columns
     * The list returned is a list of size 2 where each row
     * represents the standard deviation of each column and the mean of each column
     * @param data the data to get the standard deviation and mean for
     * @param columns the columns to get the
     * @return
     */
    public static List<Row> stdDevMeanColumns(Dataset<Row> data, String... columns) {
        return aggregate(data, columns, new String[] {"stddev", "mean"});
    }

    /**
     * Aggregate based on an arbitrary list
     * of aggregation and grouping functions
     * @param data the dataframe to aggregate
     * @param columns the columns to aggregate
     * @param functions the functions to use
     * @return the list of rows with the aggregated statistics.
     * Each row will be a function with the desired columnar output
     * in the order in which the columns were specified.
     */
    public static List<Row> aggregate(Dataset<Row> data, String[] columns, String[] functions) {
        String[] rest = new String[columns.length - 1];
        System.arraycopy(columns, 1, rest, 0, rest.length);
        List<Row> rows = new ArrayList<>();
        for (String op : functions) {
            Map<String, String> expressions = new ListOrderedMap();
            for (String s : columns) {
                expressions.put(s, op);
            }

            //compute the aggregation based on the operation
            Dataset<Row> aggregated = data.agg(expressions);
            String[] columns2 = aggregated.columns();
            //strip out the op name and parentheses from the columns
            Map<String, String> opReplace = new TreeMap<>();
            for (String s : columns2) {
                if (s.contains("min(") || s.contains("max("))
                    opReplace.put(s, s.replace(op, "").replaceAll("[()]", ""));
                else if (s.contains("avg")) {
                    opReplace.put(s, s.replace("avg", "").replaceAll("[()]", ""));
                } else {
                    opReplace.put(s, s.replace(op, "").replaceAll("[()]", ""));
                }
            }


            //get rid of the operation name in the column
            Dataset<Row> rearranged = null;
            for (Map.Entry<String, String> entries : opReplace.entrySet()) {
                //first column
                if (rearranged == null) {
                    rearranged = aggregated.withColumnRenamed(entries.getKey(), entries.getValue());
                }
                //rearranged is just a copy of aggregated at this point
                else
                    rearranged = rearranged.withColumnRenamed(entries.getKey(), entries.getValue());
            }

            rearranged = rearranged.select(DataFrames.toColumns(columns));
            //op
            rows.addAll(rearranged.collectAsList());
        }


        return rows;
    }


    /**
     * Scale based on min,max
     *
     * @param dataFrame the dataframe to scale
     * @param min       the minimum value
     * @param max       the maximum value
     * @return the normalized dataframe per column
     */
    public static Dataset<Row> normalize(Dataset<Row> dataFrame, double min, double max, List<String> skipColumns) {
        List<String> columnsList = DataFrames.toList(dataFrame.columns());
        columnsList.removeAll(skipColumns);
        String[] columnNames = DataFrames.toArray(columnsList);
        //first row is min second row is max, each column in a row is for a particular column
        List<Row> minMax = minMaxColumns(dataFrame, columnNames);
        for (int i = 0; i < columnNames.length; i++) {
            String columnName = columnNames[i];
            double dMin = ((Number) minMax.get(0).get(i)).doubleValue();
            double dMax = ((Number) minMax.get(1).get(i)).doubleValue();
            double maxSubMin = (dMax - dMin);
            if (maxSubMin == 0)
                maxSubMin = 1;

            Column newCol = dataFrame.col(columnName).minus(dMin).divide(maxSubMin).multiply(max - min).plus(min);
            dataFrame = dataFrame.withColumn(columnName, newCol);
        }


        return dataFrame;
    }

    /**
     * Scale based on min,max
     *
     * @param schema the schema of the data to scale
     * @param data   the data to scale
     * @param min    the minimum value
     * @param max    the maximum value
     * @return the normalized ata
     */
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data, double min, double max,
                    List<String> skipColumns) {
        Dataset<Row> frame = DataFrames.toDataFrame(schema, data);
        return DataFrames.toRecords(normalize(frame, min, max, skipColumns)).getSecond();
    }

    /**
     *
     * @param schema
     * @param data
     * @return
     */
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
    public static JavaRDD<List<List<Writable>>> normalizeSequence(Schema schema, JavaRDD<List<List<Writable>>> data,
                    double min, double max) {
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
        Dataset<Row> frame = DataFrames.toDataFrameSequence(schema, data);
        return DataFrames.toRecordsSequence(normalize(frame, min, max, excludeColumns)).getSecond();
    }


    /**
     * Scale based on min,max
     *
     * @param dataFrame the dataframe to scale
     * @return the normalized dataframe per column
     */
    public static Dataset<Row> normalize(Dataset<Row> dataFrame, List<String> skipColumns) {
        return normalize(dataFrame, 0, 1, skipColumns);
    }

    /**
     * Scale all data  0 to 1
     *
     * @param schema the schema of the data to scale
     * @param data   the data to scale
     * @return the normalized ata
     */
    public static JavaRDD<List<Writable>> normalize(Schema schema, JavaRDD<List<Writable>> data,
                    List<String> skipColumns) {
        return normalize(schema, data, 0, 1, skipColumns);
    }
}
