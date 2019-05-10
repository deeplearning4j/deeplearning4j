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

package org.datavec.api.transform;

import lombok.Data;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.ConvertFromSequence;
import org.datavec.api.transform.sequence.ConvertToSequence;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.io.Serializable;

/** A helper class used in TransformProcess
 * to store the types of action to
 * execute next.
 *
 * @author Alex Black
 * */
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class DataAction implements Serializable {

    private Transform transform;
    private Filter filter;
    private ConvertToSequence convertToSequence;
    private ConvertFromSequence convertFromSequence;
    private SequenceSplit sequenceSplit;
    private IAssociativeReducer reducer;
    private CalculateSortedRank calculateSortedRank;

    public DataAction() {
        //No-arg constructor for Jackson
    }

    public DataAction(Transform transform) {
        this.transform = transform;
    }

    public DataAction(Filter filter) {
        this.filter = filter;
    }

    public DataAction(ConvertToSequence convertToSequence) {
        this.convertToSequence = convertToSequence;
    }

    public DataAction(ConvertFromSequence convertFromSequence) {
        this.convertFromSequence = convertFromSequence;
    }

    public DataAction(SequenceSplit sequenceSplit) {
        this.sequenceSplit = sequenceSplit;
    }

    public DataAction(IAssociativeReducer reducer) {
        this.reducer = reducer;
    }

    public DataAction(CalculateSortedRank calculateSortedRank) {
        this.calculateSortedRank = calculateSortedRank;
    }

    @Override
    public String toString() {
        String str;
        if (transform != null) {
            str = transform.toString();
        } else if (filter != null) {
            str = filter.toString();
        } else if (convertToSequence != null) {
            str = convertToSequence.toString();
        } else if (convertFromSequence != null) {
            str = convertFromSequence.toString();
        } else if (sequenceSplit != null) {
            str = sequenceSplit.toString();
        } else if (reducer != null) {
            str = reducer.toString();
        } else if (calculateSortedRank != null) {
            str = calculateSortedRank.toString();
        } else {
            throw new IllegalStateException(
                            "Invalid DataAction: does not contain any operation to perform (all fields are null)");
        }
        return "DataAction(" + str + ")";
    }

    public Schema getSchema() {
        if (transform != null) {
            return transform.getInputSchema();
        } else if (filter != null) {
            return filter.getInputSchema();
        } else if (convertToSequence != null) {
            return convertToSequence.getInputSchema();
        } else if (convertFromSequence != null) {
            return convertFromSequence.getInputSchema();
        } else if (sequenceSplit != null) {
            return sequenceSplit.getInputSchema();
        } else if (reducer != null) {
            return reducer.getInputSchema();
        } else if (calculateSortedRank != null) {
            return calculateSortedRank.getInputSchema();
        } else {
            throw new IllegalStateException(
                            "Invalid DataAction: does not contain any operation to perform (all fields are null)");
        }
    }

}
