/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.nd4j.linalg.primitives.Pair;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import org.datavec.spark.transform.sparkfunction.SequenceToRows;
import org.datavec.spark.transform.sparkfunction.ToRecord;
import org.datavec.spark.transform.sparkfunction.ToRow;
import org.datavec.spark.transform.sparkfunction.sequence.DataFrameToSequenceCreateCombiner;
import org.datavec.spark.transform.sparkfunction.sequence.DataFrameToSequenceMergeCombiner;
import org.datavec.spark.transform.sparkfunction.sequence.DataFrameToSequenceMergeValue;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.datavec.spark.transform.DataRowsFacade.dataRows;


/**
 * Namespace for datavec
 * dataframe interop
 *
 * @author Adam Gibson
 */
public class DataFrames {

    public static final String SEQUENCE_UUID_COLUMN = "__SEQ_UUID";
    public static final String SEQUENCE_INDEX_COLUMN = "__SEQ_IDX";

    private DataFrames() {}

    /**
     * Standard deviation for a column
     *
     * @param dataFrame  the dataframe to
     *                   get the column from
     * @param columnName the name of the column to get the standard
     *                   deviation for
     * @return the column that represents the standard deviation
     */
    public static Column std(DataRowsFacade dataFrame, String columnName) {
        return functions.sqrt(var(dataFrame, columnName));
    }


    /**
     * Standard deviation for a column
     *
     * @param dataFrame  the dataframe to
     *                   get the column from
     * @param columnName the name of the column to get the standard
     *                   deviation for
     * @return the column that represents the standard deviation
     */
    public static Column var(DataRowsFacade dataFrame, String columnName) {
        return dataFrame.get().groupBy(columnName).agg(functions.variance(columnName)).col(columnName);
    }

    /**
     * MIn for a column
     *
     * @param dataFrame  the dataframe to
     *                   get the column from
     * @param columnName the name of the column to get the min for
     * @return the column that represents the min
     */
    public static Column min(DataRowsFacade dataFrame, String columnName) {
        return dataFrame.get().groupBy(columnName).agg(functions.min(columnName)).col(columnName);
    }

    /**
     * Max for a column
     *
     * @param dataFrame  the dataframe to
     *                   get the column from
     * @param columnName the name of the column
     *                   to get the max for
     * @return the column that represents the max
     */
    public static Column max(DataRowsFacade dataFrame, String columnName) {
        return dataFrame.get().groupBy(columnName).agg(functions.max(columnName)).col(columnName);
    }

    /**
     * Mean for a column
     *
     * @param dataFrame  the dataframe to
     *                   get the column fron
     * @param columnName the name of the column to get the mean for
     * @return the column that represents the mean
     */
    public static Column mean(DataRowsFacade dataFrame, String columnName) {
        return dataFrame.get().groupBy(columnName).agg(avg(columnName)).col(columnName);
    }

    /**
     * Convert a datavec schema to a
     * struct type in spark
     *
     * @param schema the schema to convert
     * @return the datavec struct type
     */
    public static StructType fromSchema(Schema schema) {
        StructField[] structFields = new StructField[schema.numColumns()];
        for (int i = 0; i < structFields.length; i++) {
            switch (schema.getColumnTypes().get(i)) {
                case Double:
                    structFields[i] = new StructField(schema.getName(i), DataTypes.DoubleType, false, Metadata.empty());
                    break;
                case Integer:
                    structFields[i] =
                                    new StructField(schema.getName(i), DataTypes.IntegerType, false, Metadata.empty());
                    break;
                case Long:
                    structFields[i] = new StructField(schema.getName(i), DataTypes.LongType, false, Metadata.empty());
                    break;
                case Float:
                    structFields[i] = new StructField(schema.getName(i), DataTypes.FloatType, false, Metadata.empty());
                    break;
                default:
                    throw new IllegalStateException(
                                    "This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
            }
        }
        return new StructType(structFields);
    }

    /**
     * Convert the DataVec sequence schema to a StructType for Spark, for example for use in
     * {@link #toDataFrameSequence(Schema, JavaRDD)}}
     * <b>Note</b>: as per {@link #toDataFrameSequence(Schema, JavaRDD)}}, the StructType has two additional columns added to it:<br>
     * - Column 0: Sequence UUID (name: {@link #SEQUENCE_UUID_COLUMN}) - a UUID for the original sequence<br>
     * - Column 1: Sequence index (name: {@link #SEQUENCE_INDEX_COLUMN} - an index (integer, starting at 0) for the position
     * of this record in the original time series.<br>
     * These two columns are required if the data is to be converted back into a sequence at a later point, for example
     * using {@link #toRecordsSequence(DataRowsFacade)}
     *
     * @param schema Schema to convert
     * @return StructType for the schema
     */
    public static StructType fromSchemaSequence(Schema schema) {
        StructField[] structFields = new StructField[schema.numColumns() + 2];

        structFields[0] = new StructField(SEQUENCE_UUID_COLUMN, DataTypes.StringType, false, Metadata.empty());
        structFields[1] = new StructField(SEQUENCE_INDEX_COLUMN, DataTypes.IntegerType, false, Metadata.empty());

        for (int i = 0; i < schema.numColumns(); i++) {
            switch (schema.getColumnTypes().get(i)) {
                case Double:
                    structFields[i + 2] =
                                    new StructField(schema.getName(i), DataTypes.DoubleType, false, Metadata.empty());
                    break;
                case Integer:
                    structFields[i + 2] =
                                    new StructField(schema.getName(i), DataTypes.IntegerType, false, Metadata.empty());
                    break;
                case Long:
                    structFields[i + 2] =
                                    new StructField(schema.getName(i), DataTypes.LongType, false, Metadata.empty());
                    break;
                case Float:
                    structFields[i + 2] =
                                    new StructField(schema.getName(i), DataTypes.FloatType, false, Metadata.empty());
                    break;
                default:
                    throw new IllegalStateException(
                                    "This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
            }
        }
        return new StructType(structFields);
    }


    /**
     * Create a datavec schema
     * from a struct type
     *
     * @param structType the struct type to create the schema from
     * @return the created schema
     */
    public static Schema fromStructType(StructType structType) {
        Schema.Builder builder = new Schema.Builder();
        StructField[] fields = structType.fields();
        String[] fieldNames = structType.fieldNames();
        for (int i = 0; i < fields.length; i++) {
            String name = fields[i].dataType().typeName().toLowerCase();
            switch (name) {
                case "double":
                    builder.addColumnDouble(fieldNames[i]);
                    break;
                case "float":
                    builder.addColumnFloat(fieldNames[i]);
                    break;
                case "long":
                    builder.addColumnLong(fieldNames[i]);
                    break;
                case "int":
                case "integer":
                    builder.addColumnInteger(fieldNames[i]);
                    break;
                case "string":
                    builder.addColumnString(fieldNames[i]);
                    break;
                default:
                    throw new RuntimeException("Unknown type: " + name);
            }
        }

        return builder.build();
    }


    /**
     * Create a compatible schema
     * and rdd for datavec
     *
     * @param dataFrame the dataframe to convert
     * @return the converted schema and rdd of writables
     */
    public static Pair<Schema, JavaRDD<List<Writable>>> toRecords(DataRowsFacade dataFrame) {
        Schema schema = fromStructType(dataFrame.get().schema());
        return new Pair<>(schema, dataFrame.get().javaRDD().map(new ToRecord(schema)));
    }

    /**
     * Convert the given DataFrame to a sequence<br>
     * <b>Note</b>: It is assumed here that the DataFrame has been created by {@link #toDataFrameSequence(Schema, JavaRDD)}.
     * In particular:<br>
     * - the first column is a UUID for the original sequence the row is from<br>
     * - the second column is a time step index: where the row appeared in the original sequence<br>
     * <p>
     * Typical use: Normalization via the {@link Normalization} static methods
     *
     * @param dataFrame Data frame to convert
     * @return Data in sequence (i.e., {@code List<List<Writable>>} form
     */
    public static Pair<Schema, JavaRDD<List<List<Writable>>>> toRecordsSequence(DataRowsFacade dataFrame) {

        //Need to convert from flattened to sequence data...
        //First: Group by the Sequence UUID (first column)
        JavaPairRDD<String, Iterable<Row>> grouped = dataFrame.get().javaRDD().groupBy(new Function<Row, String>() {
            @Override
            public String call(Row row) throws Exception {
                return row.getString(0);
            }
        });


        Schema schema = fromStructType(dataFrame.get().schema());

        //Group by sequence UUID, and sort each row within the sequences using the time step index
        Function<Iterable<Row>, List<List<Writable>>> createCombiner = new DataFrameToSequenceCreateCombiner(schema); //Function to create the initial combiner
        Function2<List<List<Writable>>, Iterable<Row>, List<List<Writable>>> mergeValue =
                        new DataFrameToSequenceMergeValue(schema); //Function to add a row
        Function2<List<List<Writable>>, List<List<Writable>>, List<List<Writable>>> mergeCombiners =
                        new DataFrameToSequenceMergeCombiner(); //Function to merge existing sequence writables

        JavaRDD<List<List<Writable>>> sequences =
                        grouped.combineByKey(createCombiner, mergeValue, mergeCombiners).values();

        //We no longer want/need the sequence UUID and sequence time step columns - extract those out
        JavaRDD<List<List<Writable>>> out = sequences.map(new Function<List<List<Writable>>, List<List<Writable>>>() {
            @Override
            public List<List<Writable>> call(List<List<Writable>> v1) throws Exception {
                List<List<Writable>> out = new ArrayList<>(v1.size());
                for (List<Writable> l : v1) {
                    List<Writable> subset = new ArrayList<>();
                    for (int i = 2; i < l.size(); i++) {
                        subset.add(l.get(i));
                    }
                    out.add(subset);
                }
                return out;
            }
        });

        return new Pair<>(schema, out);
    }

    /**
     * Creates a data frame from a collection of writables
     * rdd given a schema
     *
     * @param schema the schema to use
     * @param data   the data to convert
     * @return the dataframe object
     */
    public static DataRowsFacade toDataFrame(Schema schema, JavaRDD<List<Writable>> data) {
        JavaSparkContext sc = new JavaSparkContext(data.context());
        SQLContext sqlContext = new SQLContext(sc);
        JavaRDD<Row> rows = data.map(new ToRow(schema));
        return dataRows(sqlContext.createDataFrame(rows, fromSchema(schema)));
    }


    /**
     * Convert the given sequence data set to a DataFrame.<br>
     * <b>Note</b>: The resulting DataFrame has two additional columns added to it:<br>
     * - Column 0: Sequence UUID (name: {@link #SEQUENCE_UUID_COLUMN}) - a UUID for the original sequence<br>
     * - Column 1: Sequence index (name: {@link #SEQUENCE_INDEX_COLUMN} - an index (integer, starting at 0) for the position
     * of this record in the original time series.<br>
     * These two columns are required if the data is to be converted back into a sequence at a later point, for example
     * using {@link #toRecordsSequence(DataRowsFacade)}
     *
     * @param schema Schema for the data
     * @param data   Sequence data to convert to a DataFrame
     * @return The dataframe object
     */
    public static DataRowsFacade toDataFrameSequence(Schema schema, JavaRDD<List<List<Writable>>> data) {
        JavaSparkContext sc = new JavaSparkContext(data.context());

        SQLContext sqlContext = new SQLContext(sc);
        JavaRDD<Row> rows = data.flatMap(new SequenceToRows(schema));
        return dataRows(sqlContext.createDataFrame(rows, fromSchemaSequence(schema)));
    }

    /**
     * Convert a given Row to a list of writables, given the specified Schema
     *
     * @param schema Schema for the data
     * @param row    Row of data
     */
    public static List<Writable> rowToWritables(Schema schema, Row row) {
        List<Writable> ret = new ArrayList<>();
        for (int i = 0; i < row.size(); i++) {
            switch (schema.getType(i)) {
                case Double:
                    ret.add(new DoubleWritable(row.getDouble(i)));
                    break;
                case Float:
                    ret.add(new FloatWritable(row.getFloat(i)));
                    break;
                case Integer:
                    ret.add(new IntWritable(row.getInt(i)));
                    break;
                case Long:
                    ret.add(new LongWritable(row.getLong(i)));
                    break;
                case String:
                    ret.add(new Text(row.getString(i)));
                    break;
                default:
                    throw new IllegalStateException("Illegal type");
            }
        }
        return ret;
    }

    /**
     * Convert a string array into a list
     * @param input the input to create the list from
     * @return the created array
     */
    public static List<String> toList(String[] input) {
        List<String> ret = new ArrayList<>();
        for (int i = 0; i < input.length; i++)
            ret.add(input[i]);
        return ret;
    }


    /**
     * Convert a string list into a array
     * @param list the input to create the array from
     * @return the created list
     */
    public static String[] toArray(List<String> list) {
        String[] ret = new String[list.size()];
        for (int i = 0; i < ret.length; i++)
            ret[i] = list.get(i);
        return ret;
    }

    /**
     * Convert a list of rows to a matrix
     * @param rows the list of rows to convert
     * @return the converted matrix
     */
    public static INDArray toMatrix(List<Row> rows) {
        INDArray ret = Nd4j.create(rows.size(), rows.get(0).size());
        for (int i = 0; i < ret.rows(); i++) {
            for (int j = 0; j < ret.columns(); j++) {
                ret.putScalar(i, j, rows.get(i).getDouble(j));
            }
        }
        return ret;
    }


    /**
     * Convert a list of string names
     * to columns
     * @param columns the columns to convert
     * @return the resulting column list
     */
    public static List<Column> toColumn(List<String> columns) {
        List<Column> ret = new ArrayList<>();
        for (String s : columns)
            ret.add(col(s));
        return ret;
    }

    /**
     * Convert an array of strings
     * to column names
     * @param columns the columns to convert
     * @return the converted columns
     */
    public static Column[] toColumns(String... columns) {
        Column[] ret = new Column[columns.length];
        for (int i = 0; i < columns.length; i++)
            ret[i] = col(columns[i]);
        return ret;
    }

}
