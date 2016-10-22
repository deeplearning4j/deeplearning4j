/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.spark.transform.sparkfunction;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.types.StructType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.DataFrames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Convert a record to a row
 * @author Adam Gibson
 */
public class SequenceToRows implements FlatMapFunction<List<List<Writable>>,Row> {
    private Schema schema;
    private StructType structType;

    public SequenceToRows(Schema schema) {
        this.schema = schema;
        structType = DataFrames.fromSchemaSequence(schema);
    }

    @Override
    public Iterable<Row> call(List<List<Writable>> sequence) throws Exception {
        if(sequence.size() == 0) return Collections.emptyList();

        String sequenceUUID = UUID.randomUUID().toString();

        List<Row> out = new ArrayList<>(sequence.size());

        int stepCount = 0;
        for(List<Writable> step : sequence ) {
            Object[] values = new Object[step.size() + 2];
            values[0] = sequenceUUID;
            values[1] = stepCount++;
            for (int i = 0; i < step.size(); i++) {
                switch (schema.getColumnTypes().get(i)) {
                    case Double:
                        values[i+2] = step.get(i).toDouble();
                        break;
                    case Integer:
                        values[i+2] = step.get(i).toInt();
                        break;
                    case Long:
                        values[i+2] = step.get(i).toLong();
                        break;
                    case Float:
                        values[i+2] = step.get(i).toFloat();
                        break;
                    default:
                        throw new IllegalStateException("This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
                }
            }

            Row row = new GenericRowWithSchema(values,structType);
            out.add(row);
        }


        return out;
    }
}
