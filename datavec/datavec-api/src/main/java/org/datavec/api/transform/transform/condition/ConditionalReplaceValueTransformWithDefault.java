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

package org.datavec.api.transform.transform.condition;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnOp;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Replace the value in a specified column with a 'yes' value, if a condition is satisfied/true.<br>
 * Replace the value of this same column with a 'no' value otherwise.
 * Note that the condition can be any generic condition, including on other column(s), different to the column
 * that will be modified if the condition is satisfied/true.<br>
 * <p>
 * <b>Note</b>: For sequences, this transform use the convention that each step in the sequence is passed to the condition,
 * and replaced (or not) separately (i.e., Condition.condition(List<Writable>) is used on each time step individually)
 *
 * @author Alex Black
 * @author kepricon
 * @see ConditionalReplaceValueTransform the version without a 'no' Value
 */
@JsonIgnoreProperties({"filterColIdx"})
@EqualsAndHashCode(exclude = {"filterColIdx"})
@Data
public class ConditionalReplaceValueTransformWithDefault implements Transform, ColumnOp {


    protected final String columnToReplace;
    protected Writable yesVal;
    protected Writable noVal;
    protected int filterColIdx;
    protected final Condition condition;

    public ConditionalReplaceValueTransformWithDefault(@JsonProperty("columnToReplace") String columnToReplace,
                                @JsonProperty("yesVal") Writable yesVal,
                                @JsonProperty("noVal") Writable noVal,
                                @JsonProperty("condiiton") Condition condition) {
        this.columnToReplace = columnToReplace;
        this.yesVal = yesVal;
        this.noVal = noVal;
        this.condition = condition;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        //Conditional replace should not change any of the metadata, under normal usage
        return inputSchema;
    }

    @Override
    public void setInputSchema(Schema inputSchema){
        this.filterColIdx = inputSchema.getColumnNames().indexOf(columnToReplace);
        if (this.filterColIdx < 0) {
            throw new IllegalStateException("Column \"" + columnToReplace + "\" not found in input schema");
        }
        condition.setInputSchema(inputSchema);
    }


    @Override
    public Schema getInputSchema() {
        return condition.getInputSchema();
    }

    @Override
    public String outputColumnName() {
        return columnToReplace;
    }

    @Override
    public String[] outputColumnNames() {
        return columnNames();
    }

    @Override
    public String[] columnNames() {
        return new String[] {columnToReplace};
    }

    @Override
    public String columnName() {
        return columnToReplace;
    }

    @Override
    public String toString() {
        return "ConditionalReplaceValueTransformWithDefault(replaceColumn=\"" + columnToReplace
            + "\",yesValue=" + yesVal
            + "\",noValue=" + noVal
            + ",condition=" + condition + ")";
    }


    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>();
        for (List<Writable> step : sequence) {
            out.add(map(step));
        }
        return out;
    }

    @Override
    public Object map(Object input) {
        if (condition.condition(input)){
            return yesVal;
        } else {
            return noVal;
        }
    }

    @Override
    public Object mapSequence(Object sequence) {
        List<?> seq = (List<?>) sequence;
        List<Object> out = new ArrayList<>();
        for (Object step : seq) {
            out.add(map(step));
        }
        return out;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (condition.condition(writables)) {
            //Condition holds -> set yes value
            List<Writable> newList = new ArrayList<>(writables);
            newList.set(filterColIdx, yesVal);
            return newList;
        } else {
            //Condition does not hold -> set no value
            List<Writable> newList = new ArrayList<>(writables);
            newList.set(filterColIdx, noVal);
            return newList;
        }
    }


}
