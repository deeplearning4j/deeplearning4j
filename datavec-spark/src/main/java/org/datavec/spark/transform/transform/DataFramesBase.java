package org.datavec.spark.transform.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import org.datavec.spark.transform.DataFrames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.col;

public class DataFramesBase {

    public static final String SEQUENCE_UUID_COLUMN = "__SEQ_UUID";
    public static final String SEQUENCE_INDEX_COLUMN = "__SEQ_IDX";

    protected DataFramesBase() {
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
                    structFields[i] = new StructField(schema.getName(i), DataTypes.IntegerType, false, Metadata.empty());
                    break;
                case Long:
                    structFields[i] = new StructField(schema.getName(i), DataTypes.LongType, false, Metadata.empty());
                    break;
                case Float:
                    structFields[i] = new StructField(schema.getName(i), DataTypes.FloatType, false, Metadata.empty());
                    break;
                default:
                    throw new IllegalStateException("This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
            }
        }
        return new StructType(structFields);
    }


    /**
     * Convert the DataVec sequence schema to a StructType for Spark, for example for use in
     * {@link DataFrames#toDataFrameSequence(Schema, JavaRDD)}}
     * <b>Note</b>: as per {@link DataFrames#toDataFrameSequence(Schema, JavaRDD)}}, the StructType has two additional columns added to it:<br>
     * - Column 0: Sequence UUID (name: {@link #SEQUENCE_UUID_COLUMN}) - a UUID for the original sequence<br>
     * - Column 1: Sequence index (name: {@link #SEQUENCE_INDEX_COLUMN} - an index (integer, starting at 0) for the position
     * of this record in the original time series.<br>
     * These two columns are required if the data is to be converted back into a sequence at a later point, for example
     * using {@link DataFrames#toRecordsSequence}
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
                    structFields[i + 2] = new StructField(schema.getName(i), DataTypes.DoubleType, false, Metadata.empty());
                    break;
                case Integer:
                    structFields[i + 2] = new StructField(schema.getName(i), DataTypes.IntegerType, false, Metadata.empty());
                    break;
                case Long:
                    structFields[i + 2] = new StructField(schema.getName(i), DataTypes.LongType, false, Metadata.empty());
                    break;
                case Float:
                    structFields[i + 2] = new StructField(schema.getName(i), DataTypes.FloatType, false, Metadata.empty());
                    break;
                default:
                    throw new IllegalStateException("This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
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
        for(int i = 0; i < input.length; i++)
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
        for(int i = 0; i < ret.length; i++)
            ret[i] = list.get(i);
        return ret;
    }

    /**
     * Convert a list of rows to a matrix
     * @param rows the list of rows to convert
     * @return the converted matrix
     */
    public static INDArray toMatrix(List<Row> rows) {
        INDArray ret = Nd4j.create(rows.size(),rows.get(0).size());
        for(int i = 0; i < ret.rows(); i++) {
            for(int j = 0; j < ret.columns(); j++) {
                ret.putScalar(i,j,rows.get(i).getDouble(j));
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
        for(String s : columns)
            ret.add(col(s));
        return ret;
    }
    /**
     * Convert an array of strings
     * to column names
     * @param columns the columns to convert
     * @return the converted columns
     */
    public static Column[] toColumns(String...columns) {
        Column[] ret = new Column[columns.length];
        for(int i = 0; i < columns.length; i++)
            ret[i] = col(columns[i]);
        return ret;
    }
}
