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

package org.datavec.api.transform.reduce.geo;

import lombok.Getter;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.ops.IAggregableReduceOp;
import org.datavec.api.transform.reduce.AggregableColumnReduction;
import org.datavec.api.transform.reduce.AggregableReductionUtils;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Supplier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Applies a ReduceOp to a column of coordinates, for each component independently.
 * Basically a dispatchop with n = 2 an integrated coordinate parsing & serialization
 *
 * @author saudet
 */
public class CoordinatesReduction implements AggregableColumnReduction {
    public static final String DEFAULT_COLUMN_NAME = "CoordinatesReduction";

    public final static String DEFAULT_DELIMITER = ":";
    protected String delimiter = DEFAULT_DELIMITER;

    private final List<String> columnNamesPostReduce;

    private final Supplier<IAggregableReduceOp<Writable, List<Writable>>> multiOp(final List<ReduceOp> ops) {
        return new Supplier<IAggregableReduceOp<Writable, List<Writable>>>() {
            @Override
            public IAggregableReduceOp<Writable, List<Writable>> get() {
                return AggregableReductionUtils.reduceDoubleColumn(ops, false, null);
            }
        };
    }

    public CoordinatesReduction(String columnNamePostReduce, ReduceOp op) {
        this(columnNamePostReduce, op, DEFAULT_DELIMITER);
    }

    public CoordinatesReduction(List<String> columnNamePostReduce, List<ReduceOp> op) {
        this(columnNamePostReduce, op, DEFAULT_DELIMITER);
    }

    public CoordinatesReduction(String columnNamePostReduce, ReduceOp op, String delimiter) {
        this(Collections.singletonList(columnNamePostReduce), Collections.singletonList(op), delimiter);
    }

    public CoordinatesReduction(List<String> columnNamesPostReduce, List<ReduceOp> ops, String delimiter) {
        this.columnNamesPostReduce = columnNamesPostReduce;
        this.reducer = new CoordinateAggregableReduceOp(ops.size(), multiOp(ops), delimiter);
    }

    @Override
    public List<String> getColumnsOutputName(String columnInputName) {
        return columnNamesPostReduce;
    }

    @Override
    public List<ColumnMetaData> getColumnOutputMetaData(List<String> newColumnName, ColumnMetaData columnInputMeta) {
        List<ColumnMetaData> res = new ArrayList<>(newColumnName.size());
        for (String cn : newColumnName)
            res.add(new StringMetaData((cn)));
        return res;
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

    private IAggregableReduceOp<Writable, List<Writable>> reducer;

    @Override
    public IAggregableReduceOp<Writable, List<Writable>> reduceOp() {
        return reducer;
    }


    public static class CoordinateAggregableReduceOp implements IAggregableReduceOp<Writable, List<Writable>> {


        private int nOps;
        private Supplier<IAggregableReduceOp<Writable, List<Writable>>> initialOpValue;
        @Getter
        private ArrayList<IAggregableReduceOp<Writable, List<Writable>>> perCoordinateOps; // of size coords()
        private String delimiter;

        public CoordinateAggregableReduceOp(int n, Supplier<IAggregableReduceOp<Writable, List<Writable>>> initialOp,
                        String delim) {
            this.nOps = n;
            this.perCoordinateOps = new ArrayList<>();
            this.initialOpValue = initialOp;
            this.delimiter = delim;
        }

        @Override
        public <W extends IAggregableReduceOp<Writable, List<Writable>>> void combine(W accu) {
            if (accu instanceof CoordinateAggregableReduceOp) {
                CoordinateAggregableReduceOp accumulator = (CoordinateAggregableReduceOp) accu;
                for (int i = 0; i < Math.min(perCoordinateOps.size(), accumulator.getPerCoordinateOps().size()); i++) {
                    perCoordinateOps.get(i).combine(accumulator.getPerCoordinateOps().get(i));
                } // the rest is assumed identical
            }
        }

        @Override
        public void accept(Writable writable) {
            String[] coordinates = writable.toString().split(delimiter);
            for (int i = 0; i < coordinates.length; i++) {
                String coordinate = coordinates[i];
                while (perCoordinateOps.size() < i + 1) {
                    perCoordinateOps.add(initialOpValue.get());
                }
                perCoordinateOps.get(i).accept(new DoubleWritable(Double.parseDouble(coordinate)));
            }
        }

        @Override
        public List<Writable> get() {
            List<StringBuilder> res = new ArrayList<>(nOps);
            for (int i = 0; i < nOps; i++) {
                res.add(new StringBuilder());
            }

            for (int i = 0; i < perCoordinateOps.size(); i++) {
                List<Writable> resThisCoord = perCoordinateOps.get(i).get();
                for (int j = 0; j < nOps; j++) {
                    res.get(j).append(resThisCoord.get(j).toString());
                    if (i < perCoordinateOps.size() - 1) {
                        res.get(j).append(delimiter);
                    }
                }
            }

            List<Writable> finalRes = new ArrayList<>(nOps);
            for (StringBuilder sb : res) {
                finalRes.add(new Text(sb.toString()));
            }
            return finalRes;
        }
    }

}
