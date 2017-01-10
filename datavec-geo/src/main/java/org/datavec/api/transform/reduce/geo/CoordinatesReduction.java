/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.transform.reduce.geo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.reduce.ColumnReduction;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

/**
 * Applies a ReduceOp to a column of coordinates, for each component independently.
 *
 * @author saudet
 */
public class CoordinatesReduction implements ColumnReduction {
    public static final String DEFAULT_COLUMN_NAME = "CoordinatesReduction";

    public final static String DEFAULT_DELIMITER = ":";
    protected String delimiter = DEFAULT_DELIMITER;

    private final String columnNamePostReduce;
    private final ReduceOp op;

    public CoordinatesReduction(String columnNamePostReduce, ReduceOp op) {
        this(columnNamePostReduce, op, DEFAULT_DELIMITER);
    }

    public CoordinatesReduction(String columnNamePostReduce, ReduceOp op, String delimiter) {
        this.columnNamePostReduce = columnNamePostReduce;
        this.op = op;
        this.delimiter = delimiter;
    }

    @Override
    public Writable reduceColumn(List<Writable> columnData) {
        ArrayList<Writable>[] values = new ArrayList[1];
        for (Writable w : columnData) {
            String[] coordinates = w.toString().split(delimiter);
            if (values.length < coordinates.length) {
                values = Arrays.copyOf(values, coordinates.length);
            }
            for (int i = 0; i < coordinates.length; i++) {
                String coordinate = coordinates[i];
                if (values[i] == null) {
                    values[i] = new ArrayList<Writable>(columnData.size());
                }
                values[i].add(new DoubleWritable(Double.parseDouble(coordinate)));
            }
        }
        String output = "";
        for (int i = 0; i < values.length; i++) {
            output += Reducer.reduceDoubleColumn(op, values[i], false, null).toString();
            if (i < values.length - 1) {
                output += delimiter;
            }
        }
        return new Text(output);
    }

    @Override
    public String getColumnOutputName(String columnInputName) {
        return columnNamePostReduce;
    }

    @Override
    public ColumnMetaData getColumnOutputMetaData(String newColumnName, ColumnMetaData columnInputMeta) {
        return new StringMetaData(newColumnName);
    }

    @Override
    public Schema transform(Schema inputSchema) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Schema getInputSchema() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String outputColumnName() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String[] outputColumnNames() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String[] columnNames() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String columnName() {
        throw new UnsupportedOperationException();
    }
}
