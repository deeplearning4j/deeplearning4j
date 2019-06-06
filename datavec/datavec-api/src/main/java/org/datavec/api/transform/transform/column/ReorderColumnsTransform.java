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

package org.datavec.api.transform.transform.column;

import lombok.Data;
import org.datavec.api.transform.ColumnOp;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Rearrange the order of the columns.
 * Note: A partial list of columns can be used here. Any columns that are not explicitly mentioned
 * will be placed after those that are in the output, without changing their relative order.
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema", "outputOrder"})
@Data
public class ReorderColumnsTransform implements Transform, ColumnOp {

    private final List<String> newOrder;
    private Schema inputSchema;
    private int[] outputOrder; //Mapping from in to out. so output[i] = input.get(outputOrder[i])

    /**
     * @param newOrder A partial or complete order of the columns in the output
     */
    public ReorderColumnsTransform(String... newOrder) {
        this(Arrays.asList(newOrder));
    }

    /**
     * @param newOrder A partial or complete order of the columns in the output
     */
    public ReorderColumnsTransform(@JsonProperty("newOrder") List<String> newOrder) {
        this.newOrder = newOrder;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        for (String s : newOrder) {
            if (!inputSchema.hasColumn(s)) {
                throw new IllegalStateException("Input schema does not contain column with name \"" + s + "\"");
            }
        }
        if (inputSchema.numColumns() < newOrder.size())
            throw new IllegalArgumentException("Schema has " + inputSchema.numColumns() + " column but newOrder has "
                            + newOrder.size() + " columns");

        List<String> origNames = inputSchema.getColumnNames();
        List<ColumnMetaData> origMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> outMeta = new ArrayList<>();

        boolean[] taken = new boolean[origNames.size()];
        for (String s : newOrder) {
            int idx = inputSchema.getIndexOfColumn(s);
            outMeta.add(origMeta.get(idx));
            taken[idx] = true;
        }

        for (int i = 0; i < taken.length; i++) {
            if (taken[i])
                continue;
            outMeta.add(origMeta.get(i));
        }

        return inputSchema.newSchema(outMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        for (String s : newOrder) {
            if (!inputSchema.hasColumn(s)) {
                throw new IllegalStateException("Input schema does not contain column with name \"" + s + "\"");
            }
        }
        if (inputSchema.numColumns() < newOrder.size())
            throw new IllegalArgumentException("Schema has " + inputSchema.numColumns() + " columns but newOrder has "
                            + newOrder.size() + " columns");

        List<String> origNames = inputSchema.getColumnNames();
        outputOrder = new int[origNames.size()];

        boolean[] taken = new boolean[origNames.size()];
        int j = 0;
        for (String s : newOrder) {
            int idx = inputSchema.getIndexOfColumn(s);
            taken[idx] = true;
            outputOrder[j++] = idx;
        }

        for (int i = 0; i < taken.length; i++) {
            if (taken[i])
                continue;
            outputOrder[j++] = i;
        }
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        List<Writable> out = new ArrayList<>();
        for (int i : outputOrder) {
            out.add(writables.get(i));
        }
        return out;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>();
        for (List<Writable> step : sequence) {
            out.add(map(step));
        }
        return out;
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException(
                        "Unable to map. Please treat this as a special operation. This should be handled by your implementation.");

    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException(
                        "Unable to map. Please treat this as a special operation. This should be handled by your implementation.");
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        ReorderColumnsTransform o2 = (ReorderColumnsTransform) o;

        if (!newOrder.equals(o2.newOrder))
            return false;
        return Arrays.equals(outputOrder, o2.outputOrder);

    }

    @Override
    public int hashCode() {
        int result = newOrder.hashCode();
        result = 31 * result + Arrays.hashCode(outputOrder);
        return result;
    }

    @Override
    public String toString() {
        return "ReorderColumnsTransform(newOrder=" + newOrder + ")";

    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return outputColumnNames()[0];
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return newOrder.toArray(new String[newOrder.size()]);
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return getInputSchema().getColumnNames().toArray(new String[getInputSchema().getColumnNames().size()]);
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnNames()[0];
    }
}
