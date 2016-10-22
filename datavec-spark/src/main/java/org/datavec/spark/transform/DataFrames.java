package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.Function3;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.datavec.api.berkeley.Pair;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import org.datavec.spark.transform.sparkfunction.SequenceToRows;
import org.datavec.spark.transform.sparkfunction.ToRecord;
import org.datavec.spark.transform.sparkfunction.ToRow;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.apache.spark.sql.functions;
import static org.apache.spark.sql.functions.avg;



/**
 * Namespace for datavec
 * dataframe interop
 *
 * @author Adam Gibson
 */
public class DataFrames {

    public static final String SEQUENCE_UUID_COLUMN = "__SEQ_UUID";
    public static final String SEQUENCE_INDEX_COLUMN = "__SEQ_IDX";

    /**
     * Standard deviation for a column
     * @param dataFrame the dataframe to
     *                  get the column from
     * @param columnName the name of the column to get the standard
     *                   deviation for
     * @return the column that represents the standard deviation
     */
    public static Column std(DataFrame dataFrame, String columnName) {
        return functions.sqrt(var(dataFrame,columnName));
    }


    /**
     * Standard deviation for a column
     * @param dataFrame the dataframe to
     *                  get the column from
     * @param columnName the name of the column to get the standard
     *                   deviation for
     * @return the column that represents the standard deviation
     */
    public static Column var(DataFrame dataFrame, String columnName) {
        return dataFrame.groupBy(columnName).agg(functions.variance(columnName)).col(columnName);
    }

    /**
     * MIn for a column
     * @param dataFrame the dataframe to
     *                  get the column from
     * @param columnName the name of the column to get the min for
     * @return the column that represents the min
     */
    public static Column min(DataFrame dataFrame, String columnName) {
        return dataFrame.groupBy(columnName).agg(functions.min(columnName)).col(columnName);
    }

    /**
     * Max for a column
     * @param dataFrame the dataframe to
     *                  get the column from
     * @param columnName the name of the column
     *                   to get the max for
     * @return the column that represents the max
     */
    public static Column max(DataFrame dataFrame, String columnName) {
        return dataFrame.groupBy(columnName).agg(functions.max(columnName)).col(columnName);
    }

    /**
     * Mean for a column
     * @param dataFrame the dataframe to
     *                  get the column fron
     * @param columnName the name of the column to get the mean for
     * @return the column that represents the mean
     */
    public static Column mean(DataFrame dataFrame, String columnName) {
        return dataFrame.groupBy(columnName).agg(avg(columnName)).col(columnName);
    }

    /**
     * Convert a datavec schema to a
     * struct type in spark
     * @param schema the schema to convert
     * @return the datavec struct type
     */
    public static StructType fromSchema(Schema schema) {
        StructField[] structFields = new StructField[schema.numColumns()];
        for(int i = 0; i < structFields.length; i++) {
            switch (schema.getColumnTypes().get(i)) {
                case Double: structFields[i] = new StructField(schema.getName(i), DataTypes.DoubleType,false,Metadata.empty()); break;
                case Integer: new StructField(schema.getName(i), DataTypes.IntegerType,false,Metadata.empty()); break;
                case Long: new StructField(schema.getName(i), DataTypes.LongType,false,Metadata.empty()); break;
                case Float: new StructField(schema.getName(i), DataTypes.FloatType,false,Metadata.empty()); break;
                default: throw new IllegalStateException("This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
            }
        }
        return new StructType(structFields);
    }

    public static StructType fromSchemaSequence(Schema schema) {
        StructField[] structFields = new StructField[schema.numColumns()+2];

        structFields[0] = new StructField(SEQUENCE_UUID_COLUMN, DataTypes.StringType, false, Metadata.empty());
        structFields[1] = new StructField(SEQUENCE_INDEX_COLUMN, DataTypes.IntegerType, false, Metadata.empty());

        for(int i = 0; i < schema.numColumns(); i++) {
            switch (schema.getColumnTypes().get(i)) {
                case Double: structFields[i+2] = new StructField(schema.getName(i), DataTypes.DoubleType,false,Metadata.empty()); break;
                case Integer: structFields[i+2] = new StructField(schema.getName(i), DataTypes.IntegerType,false,Metadata.empty()); break;
                case Long: structFields[i+2] = new StructField(schema.getName(i), DataTypes.LongType,false,Metadata.empty()); break;
                case Float: structFields[i+2] = new StructField(schema.getName(i), DataTypes.FloatType,false,Metadata.empty()); break;
                default: throw new IllegalStateException("This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
            }
        }
        return new StructType(structFields);
    }


    /**
     * Create a datavec schema
     * from a struct type
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
     * @param dataFrame the dataframe to convert
     * @return the converted schema and rdd of writables
     */
    public static Pair<Schema,JavaRDD<List<Writable>>> toRecords(DataFrame dataFrame) {
        Schema schema = fromStructType(dataFrame.schema());
        return new Pair<>(schema,dataFrame.javaRDD().map(new ToRecord(schema)));
    }

    public static Pair<Schema, JavaRDD<List<List<Writable>>>> toRecordsSequence(DataFrame dataFrame){

        //Need to convert from flattened to sequence data...


        System.out.println("+++++++++++++ BEFORE JAVARDD +++++++++++++");
        JavaRDD<Row> rddTemp = dataFrame.javaRDD();

        System.out.println("+++++++++++++ BEFORE JAVARDD COLLECT +++++++++++++");
        List<Row> collected = rddTemp.collect();
        System.out.println("+++++++++++++ AFTER JAVARDD COLLECT +++++++++++++");
        System.out.println(collected);

        JavaPairRDD<String,Iterable<Row>> grouped = rddTemp.groupBy(new Function<Row, String>() {
            @Override
            public String call(Row row) throws Exception {
                System.out.println("///////// GROUP BY ////////////");
                return row.getString(0);    //First column is UUID
            }
        });

        System.out.println("------------ CONVERTING SEQUENCES -----------------");

        //Create the initial value
        final Schema schema = fromStructType(dataFrame.schema());
        System.out.println(schema);
        Function<Iterable<Row>,List<List<Writable>>> createCombiner = new Function<Iterable<Row>, List<List<Writable>>>() {
            @Override
            public List<List<Writable>> call(Iterable<Row> rows) throws Exception {
                System.out.println("///////// CREATE COMBINER ////////////");
                List<List<Writable>> retSeq = new ArrayList<>();
                for(Row v1 : rows ) {
                    List<Writable> ret = new ArrayList<>();
                    for (int i = 0; i < v1.size(); i++) {
                        switch (schema.getType(i)) {
                            case Double:
                                ret.add(new DoubleWritable(v1.getDouble(i)));
                                break;
                            case Float:
                                ret.add(new FloatWritable(v1.getFloat(i)));
                                break;
                            case Integer:
                                ret.add(new IntWritable(v1.getInt(i)));
                                break;
                            case Long:
                                ret.add(new LongWritable(v1.getLong(i)));
                                break;
                            case String:
                                ret.add(new Text(v1.getString(i)));
                                break;
                            default:
                                throw new IllegalStateException("Illegal type");
                        }

                    }
                    retSeq.add(ret);
                }
                System.out.println("///////// END - CREATE COMBINER ////////////");
                return retSeq;
            }
        };

        //Function to add a row;
        Function2<List<List<Writable>>,Iterable<Row>,List<List<Writable>>> mergeValue = new Function2<List<List<Writable>>, Iterable<Row>, List<List<Writable>>>() {
            @Override
            public List<List<Writable>> call(List<List<Writable>> seqSoFar, Iterable<Row> rows) throws Exception {


                List<List<Writable>> retSeq = new ArrayList<>(seqSoFar);
                for(Row v1 : rows ) {
                    List<Writable> ret = new ArrayList<>();
                    for (int i = 0; i < v1.size(); i++) {
                        switch (schema.getType(i)) {
                            case Double:
                                ret.add(new DoubleWritable(v1.getDouble(i)));
                                break;
                            case Float:
                                ret.add(new FloatWritable(v1.getFloat(i)));
                                break;
                            case Integer:
                                ret.add(new IntWritable(v1.getInt(i)));
                                break;
                            case Long:
                                ret.add(new LongWritable(v1.getLong(i)));
                                break;
                            case String:
                                ret.add(new Text(v1.getString(i)));
                                break;
                            default:
                                throw new IllegalStateException("Illegal type");
                        }

                    }
                    retSeq.add(ret);
                }

                Collections.sort(retSeq, new Comparator<List<Writable>>() {
                    @Override
                    public int compare(List<Writable> o1, List<Writable> o2) {
                        return Integer.compare(o1.get(1).toInt(), o2.get(1).toInt());
                    }
                });

                return retSeq;
            }
        };

        //Function to merge existing writables
        Function2<List<List<Writable>>, List<List<Writable>>, List<List<Writable>>> mergeCombiners = new Function2<List<List<Writable>>, List<List<Writable>>, List<List<Writable>>>() {
            @Override
            public List<List<Writable>> call(List<List<Writable>> v1, List<List<Writable>> v2) throws Exception {
                List<List<Writable>> out = new ArrayList<>(v1.size() + v2.size());
                out.addAll(v1);
                out.addAll(v2);
                Collections.sort(out, new Comparator<List<Writable>>() {
                    @Override
                    public int compare(List<Writable> o1, List<Writable> o2) {
                        return Integer.compare(o1.get(1).toInt(), o2.get(1).toInt());
                    }
                });
                return out;
            }
        };

        JavaPairRDD<String,List<List<Writable>>> seq = grouped.combineByKey(createCombiner, mergeValue, mergeCombiners);

        JavaRDD<List<List<Writable>>> out = seq.values().map(new Function<List<List<Writable>>, List<List<Writable>>>() {
            @Override
            public List<List<Writable>> call(List<List<Writable>> v1) throws Exception {
                List<List<Writable>> out = new ArrayList<>(v1.size());
                for(List<Writable> l : v1){
                    List<Writable> subset = new ArrayList<>();
                    for( int i=2; i<l.size(); i++ ){
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
     * @param schema the schema to use
     * @param data the data to convert
     * @return the dataframe object
     */
    public static DataFrame toDataFrame(Schema schema, JavaRDD<List<Writable>> data) {
        JavaSparkContext sc = new JavaSparkContext(data.context());
        SQLContext sqlContext = new SQLContext(sc);
        JavaRDD<Row> rows = data.map(new ToRow(schema));
        DataFrame dataFrame = sqlContext.createDataFrame(rows,fromSchema(schema));
        return dataFrame;
    }


    static DataFrame sequenceToDataFrame(Schema schema, JavaRDD<List<List<Writable>>> data){
        JavaSparkContext sc = new JavaSparkContext(data.context());

        SQLContext sqlContext = new SQLContext(sc);
        JavaRDD<Row> rows = data.flatMap(new SequenceToRows(schema));
        DataFrame dataFrame = sqlContext.createDataFrame(rows, fromSchemaSequence(schema));
        return dataFrame;
    }

}
