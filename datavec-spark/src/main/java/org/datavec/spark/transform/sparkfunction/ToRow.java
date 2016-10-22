package org.datavec.spark.transform.sparkfunction;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.types.*;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.DataFrames;

import java.util.List;

/**
 * Convert a record to a row
 * @author Adam Gibson
 */
public class ToRow implements Function<List<Writable>,Row> {
    private Schema schema;
    private StructType structType;

    public ToRow(Schema schema) {
        this.schema = schema;
        structType = DataFrames.fromSchema(schema);
    }

    @Override
    public Row call(List<Writable> v1) throws Exception {
        Object[] values = new Object[v1.size()];
        for(int i = 0; i < values.length; i++) {
            switch (schema.getColumnTypes().get(i)) {
                case Double: values[i] = v1.get(i).toDouble(); break;
                case Integer: v1.get(i).toInt(); break;
                case Long: v1.get(i).toLong(); break;
                case Float: v1.get(i).toFloat(); break;
                default: throw new IllegalStateException("This api should not be used with strings , binary dataor ndarrays. This is only for columnar data");
            }
        }

        Row row = new GenericRowWithSchema(values,structType);
        return row;
    }
}
