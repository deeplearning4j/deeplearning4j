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

package org.datavec.api.transform.analysis.quality.integer;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.IntegerQuality;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 3/6/17.
 */
public class IntegerQualityAnalysisState implements QualityAnalysisState<IntegerQualityAnalysisState> {

    @Getter
    private IntegerQuality integerQuality;
    private IntegerQualityAddFunction addFunction;
    private IntegerQualityMergeFunction mergeFunction;

    public IntegerQualityAnalysisState(IntegerMetaData integerMetaData) {
        this.integerQuality = new IntegerQuality(0, 0, 0, 0, 0);
        this.addFunction = new IntegerQualityAddFunction(integerMetaData);
        this.mergeFunction = new IntegerQualityMergeFunction();
    }

    public IntegerQualityAnalysisState add(Writable writable) {
        integerQuality = addFunction.apply(integerQuality, writable);
        return this;
    }

    public IntegerQualityAnalysisState merge(IntegerQualityAnalysisState other) {
        integerQuality = mergeFunction.apply(integerQuality, other.getIntegerQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return integerQuality;
    }
}
