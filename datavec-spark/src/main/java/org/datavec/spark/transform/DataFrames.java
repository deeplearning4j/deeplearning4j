package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.datavec.api.berkeley.Pair;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.sparkfunction.ToRecord;
import org.datavec.spark.transform.sparkfunction.ToRow;
import java.util.List;


/**
 * Namespace for datavec
 * dataframe interop
 *
 * @author Adam Gibson
 */
public class DataFrames {

    /**
     * Standard deviation for a column
     * @param dataFrame the dataframe to
     *                  get the column from
     * @param columnName the name of the column to get the standard
     *                   deviationfor
     * @return the column that represents the standard deviation
     */
    public static Column std(DataFrame dataFrame, String columnName) {
        return dataFrame.col(columnName).desc().getItem("stddev");
    }

    /**
     * MIn for a column
     * @param dataFrame the dataframe to
     *                  get the column from
     * @param columnName the name of the column to get the min for
     * @return the column that represents the mean
     */
    public static Column min(DataFrame dataFrame, String columnName) {
        return dataFrame.col(columnName).desc().getItem("min");
    }

    /**
     * Max for a column
     * @param dataFrame the dataframe to
     *                  get the column fron
     * @param columnName the name of the column to get the max for
     * @return the column that represents the mean
     */
    public static Column max(DataFrame dataFrame, String columnName) {
        return dataFrame.col(columnName).desc().getItem("max");
    }

    /**
     * Mean for a column
     * @param dataFrame the dataframe to
     *                  get the column fron
     * @param columnName the name of the column to get the mean for
     * @return the column that represents the mean
     */
    public static Column mean(DataFrame dataFrame, String columnName) {
        return dataFrame.col(columnName).desc().getItem("mean");
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


    /**
     *
     * @param structType
     * @return
     */
    public static Schema fromStructType(StructType structType) {
        Schema.Builder builder = new Schema.Builder();
        for(int i = 0; i < structType.fields().length; i++) {
            switch(structType.fields()[i].dataType().typeName()) {
                case "double": builder.addColumnDouble(structType.fieldNames()[i]); break;
                case "float": builder.addColumnFloat(structType.fieldNames()[i]); break;
                case "long": builder.addColumnLong(structType.fieldNames()[i]); break;
                case "int": builder.addColumnInteger(structType.fieldNames()[i]); break;
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

}
