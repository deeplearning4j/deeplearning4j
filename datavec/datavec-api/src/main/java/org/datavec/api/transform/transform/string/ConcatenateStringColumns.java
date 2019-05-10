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

package org.datavec.api.transform.transform.string;

import lombok.Data;
import org.datavec.api.transform.ColumnOp;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Concatenate values of one or more String columns into
 * a new String column. Retains the constituent String
 * columns so user must remove those manually, if desired.
 *
 * TODO: use new String Reduce functionality in DataVec?
 *
 * @author dave@skymind.io
 */
@JsonIgnoreProperties({"inputSchema"})
@Data
public class ConcatenateStringColumns extends BaseTransform implements ColumnOp {

    private final String newColumnName;
    private final String delimiter;
    private final List<String> columnsToConcatenate;
    private Schema inputSchema;

    /**
     * @param columnsToConcatenate A partial or complete order of the columns in the output
     */
    public ConcatenateStringColumns(String newColumnName, String delimiter, String... columnsToConcatenate) {
        this(newColumnName, delimiter, Arrays.asList(columnsToConcatenate));
    }

    /**
     * @param columnsToConcatenate A partial or complete order of the columns in the output
     */
    public ConcatenateStringColumns(@JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("delimiter") String delimiter,
                    @JsonProperty("columnsToConcatenate") List<String> columnsToConcatenate) {
        this.newColumnName = newColumnName;
        this.delimiter = delimiter;
        this.columnsToConcatenate = columnsToConcatenate;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        for (String s : columnsToConcatenate) {
            if (!inputSchema.hasColumn(s)) {
                throw new IllegalStateException("Input schema does not contain column with name \"" + s + "\"");
            }
        }

        List<ColumnMetaData> outMeta = new ArrayList<>();
        outMeta.addAll(inputSchema.getColumnMetaData());

        ColumnMetaData newColMeta = ColumnType.String.newColumnMetaData(newColumnName);
        outMeta.add(newColMeta);
        return inputSchema.newSchema(outMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        for (String s : columnsToConcatenate) {
            if (!inputSchema.hasColumn(s)) {
                throw new IllegalStateException("Input schema does not contain column with name \"" + s + "\"");
            }
        }
        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        StringBuilder newColumnText = new StringBuilder();
        List<Writable> out = new ArrayList<>(writables);
        int i = 0;
        for (String columnName : columnsToConcatenate) {
            if (i++ > 0)
                newColumnText.append(delimiter);
            int columnIdx = inputSchema.getIndexOfColumn(columnName);
            newColumnText.append(writables.get(columnIdx));
        }
        out.add(new Text(newColumnText.toString()));
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

        ConcatenateStringColumns o2 = (ConcatenateStringColumns) o;
        return delimiter.equals(o2.delimiter) && columnsToConcatenate.equals(o2.columnsToConcatenate);
    }

    @Override
    public int hashCode() {
        int result = delimiter.hashCode();
        result = 31 * result + columnsToConcatenate.hashCode();
        return result;
    }

    @Override
    public String toString() {
        return "ConcatenateStringColumns(delimiters=" + delimiter + " columnsToConcatenate=" + columnsToConcatenate
                        + ")";

    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return newColumnName;
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return new String[] {newColumnName};
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return columnsToConcatenate.toArray(new String[getInputSchema().getColumnNames().size()]);
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnsToConcatenate.get(0);
    }
}
