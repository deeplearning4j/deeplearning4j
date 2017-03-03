package org.datavec.spark.transform.sparkfunction;

import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.types.StructType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.DataFrames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Convert a record to a row
 * @author Adam Gibson
 */
public class SequenceToRowsAdapter implements FlatMapFunctionAdapter<List<List<Writable>>, Row> {

    private Schema schema;
    private StructType structType;

    public SequenceToRowsAdapter(Schema schema) {
        this.schema = schema;
        structType = DataFrames.fromSchemaSequence(schema);
    }


    @Override
    public Iterable<Row> call(List<List<Writable>> sequence) throws Exception {
        if (sequence.size() == 0)
            return Collections.emptyList();

        String sequenceUUID = UUID.randomUUID().toString();

        List<Row> out = new ArrayList<>(sequence.size());

        int stepCount = 0;
        for (List<Writable> step : sequence) {
            Object[] values = new Object[step.size() + 2];
            values[0] = sequenceUUID;
            values[1] = stepCount++;
            for (int i = 0; i < step.size(); i++) {
                switch (schema.getColumnTypes().get(i)) {
                    case Double:
                        values[i + 2] = step.get(i).toDouble();
                        break;
                    case Integer:
                        values[i + 2] = step.get(i).toInt();
                        break;
                    case Long:
                        values[i + 2] = step.get(i).toLong();
                        break;
                    case Float:
                        values[i + 2] = step.get(i).toFloat();
                        break;
                    default:
                        throw new IllegalStateException(
                                        "This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
                }
            }

            Row row = new GenericRowWithSchema(values, structType);
            out.add(row);
        }

        return out;
    }
}
