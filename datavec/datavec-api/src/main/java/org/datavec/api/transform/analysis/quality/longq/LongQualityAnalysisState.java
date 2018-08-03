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

package org.datavec.api.transform.analysis.quality.longq;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.LongMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.LongQuality;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 3/6/17.
 */
public class LongQualityAnalysisState implements QualityAnalysisState<LongQualityAnalysisState> {

    @Getter
    private LongQuality longQuality;
    private LongQualityAddFunction addFunction;
    private LongQualityMergeFunction mergeFunction;

    public LongQualityAnalysisState(LongMetaData longMetaData) {
        this.longQuality = new LongQuality();
        this.addFunction = new LongQualityAddFunction(longMetaData);
        this.mergeFunction = new LongQualityMergeFunction();
    }

    public LongQualityAnalysisState add(Writable writable){
        longQuality = addFunction.apply(longQuality, writable);
        return this;
    }

    public LongQualityAnalysisState merge(LongQualityAnalysisState other){
        longQuality = mergeFunction.apply(longQuality, other.getLongQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return longQuality;
    }
}
