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

package org.datavec.api.transform.rank;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.LongMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.serde.legacy.LegacyMappingHelper;
import org.datavec.api.writable.comparator.WritableComparator;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * CalculateSortedRank: calculate the rank of each example, after sorting example.
 * For example, we might have some numerical "score" column, and we want to know for the rank (sort order) for each
 * example, according to that column.<br>
 * The rank of each example (after sorting) will be added in a new Long column. Indexing is done from 0; examples will have
 * values 0 to dataSetSize - 1.<br>
 *
 * Currently, CalculateSortedRank can only be applied on standard (i.e., non-sequence) data.
 * Furthermore, the current implementation can only sort on one column
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"inputSchema"})
@JsonIgnoreProperties({"inputSchema"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyMappingHelper.CalculateSortedRankHelper.class)
public class CalculateSortedRank implements Serializable, ColumnOp {

    private final String newColumnName;
    private final String sortOnColumn;
    private final WritableComparator comparator;
    private final boolean ascending;
    private Schema inputSchema;


    /**
     *
     * @param newColumnName    Name of the new column (will contain the rank for each example)
     * @param sortOnColumn     Name of the column to sort on
     * @param comparator       Comparator used to sort examples
     */
    public CalculateSortedRank(String newColumnName, String sortOnColumn, WritableComparator comparator) {
        this(newColumnName, sortOnColumn, comparator, true);
    }

    /**
     *
     * @param newColumnName    Name of the new column (will contain the rank for each example)
     * @param sortOnColumn     Name of the column to sort on
     * @param comparator       Comparator used to sort examples
     * @param ascending        Whether examples should be ascending or descending, using the comparator
     */
    public CalculateSortedRank(@JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("sortOnColumn") String sortOnColumn,
                    @JsonProperty("comparator") WritableComparator comparator,
                    @JsonProperty("ascending") boolean ascending) {
        this.newColumnName = newColumnName;
        this.sortOnColumn = sortOnColumn;
        this.comparator = comparator;
        this.ascending = ascending;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        if (inputSchema instanceof SequenceSchema)
            throw new IllegalStateException("Calculating sorted rank on sequences: not yet supported");

        List<ColumnMetaData> origMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(origMeta);

        newMeta.add(new LongMetaData(newColumnName, 0L, null));

        return inputSchema.newSchema(newMeta);
    }

    @Override
    public void setInputSchema(Schema schema) {
        this.inputSchema = schema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
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
        List<String> columnNames = inputSchema.getColumnNames();
        columnNames.add(newColumnName);
        return columnNames.toArray(new String[columnNames.size()]);
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return inputSchema.getColumnNames().toArray(new String[inputSchema.numColumns()]);
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

    @Override
    public String toString() {
        return "CalculateSortedRank(newColumnName=\"" + newColumnName + "\", comparator=" + comparator + ")";
    }
}
