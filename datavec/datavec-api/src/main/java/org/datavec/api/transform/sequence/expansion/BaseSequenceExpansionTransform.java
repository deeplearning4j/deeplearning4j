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

package org.datavec.api.transform.sequence.expansion;

import lombok.*;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.*;

/**
 * A base class for sequence expansion operations.
 * The idea: for one or more columns, expand the values to multiple sequence steps; for all other columns, just
 * duplicate the step values when expanding.
 *
 * @author Alex Black
 */
@EqualsAndHashCode(exclude = {"inputSchema"})
@JsonIgnoreProperties({"inputSchema"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public abstract class BaseSequenceExpansionTransform implements Transform {

    protected List<String> requiredColumns;
    protected List<String> expandedColumnNames;

    @Setter @Getter
    protected Schema inputSchema;

    /**
     * @param requiredColumns      Input columns, to be expanded
     * @param expandedColumnNames  Names of the columns after expansion
     */
    protected BaseSequenceExpansionTransform(@NonNull List<String> requiredColumns, @NonNull List<String> expandedColumnNames){
        if(requiredColumns.size() == 0){
            throw new IllegalArgumentException("No columns have values to be expanded. Must have requiredColumns.size() > 0");
        }
        this.requiredColumns = requiredColumns;
        this.expandedColumnNames = expandedColumnNames;
    }

    protected abstract List<ColumnMetaData> expandedColumnMetaDatas(List<ColumnMetaData> origColumnMeta, List<String> expandedColumnNames);

    protected abstract List<List<Writable>> expandTimeStep(List<Writable> currentStepValues);


    @Override
    public Schema transform(Schema inputSchema) {
        //Same schema *except* for the expanded columns

        List<ColumnMetaData> meta = new ArrayList<>(inputSchema.numColumns());

        List<ColumnMetaData> oldMetaToExpand = new ArrayList<>();
        for(String s : requiredColumns){
            oldMetaToExpand.add(inputSchema.getMetaData(s));
        }
        List<ColumnMetaData> newMetaToExpand = expandedColumnMetaDatas(oldMetaToExpand, expandedColumnNames);

        int modColumnIdx = 0;
        for(ColumnMetaData m : inputSchema.getColumnMetaData()){

            if(requiredColumns.contains(m.getName())){
                //Possibly changed column (expanded)
                meta.add(newMetaToExpand.get(modColumnIdx++));
            } else {
                //Unmodified column
                meta.add(m);
            }
        }

        return inputSchema.newSchema(meta);
    }

    @Override
    public String outputColumnName() {
        return expandedColumnNames.get(0);
    }

    @Override
    public String[] outputColumnNames() {
        return expandedColumnNames.toArray(new String[requiredColumns.size()]);
    }

    @Override
    public String[] columnNames() {
        return requiredColumns.toArray(new String[requiredColumns.size()]);
    }

    @Override
    public String columnName() {
        return columnNames()[0];
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        throw new UnsupportedOperationException("Cannot perform sequence expansion on non-sequence data");
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {

        int nCols = inputSchema.numColumns();
        List<List<Writable>> out = new ArrayList<>();

        Map<Integer,Integer> expandColumnIdxsMap = new HashMap<>(); //Map from "position in vector idx" to "required column idx"
        int[] expandColumnIdxs = new int[requiredColumns.size()];
        int count=0;
        for( String s : requiredColumns ){
            int idx = inputSchema.getIndexOfColumn(s);
            expandColumnIdxsMap.put(idx, count);
            expandColumnIdxs[count++] = idx;
        }

        List<Writable> toExpand = new ArrayList<>(requiredColumns.size());
        for(List<Writable> step : sequence){
            toExpand.clear();

            for( int i : expandColumnIdxs ){
                toExpand.add(step.get(i));
            }

            List<List<Writable>> expanded = expandTimeStep(toExpand);

            //Now: for each expanded step, copy the original values out
            int expansionSize = expanded.size();
            for( int i=0; i<expansionSize; i++ ){
                List<Writable> newStep = new ArrayList<>(nCols);
                for( int j=0; j<nCols; j++ ){
                    if(expandColumnIdxsMap.containsKey(j)){
                        //This is one of the expanded columns
                        int expandIdx = expandColumnIdxsMap.get(j);
                        newStep.add(expanded.get(i).get(expandIdx));
                    } else {
                        //Copy existing  value
                        newStep.add(step.get(j));
                    }
                }

                out.add(newStep);
            }
        }

        return out;
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
