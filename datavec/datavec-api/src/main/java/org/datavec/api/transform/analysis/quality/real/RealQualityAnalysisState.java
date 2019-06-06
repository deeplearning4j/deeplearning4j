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

package org.datavec.api.transform.analysis.quality.real;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.DoubleQuality;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 3/6/17.
 */
public class RealQualityAnalysisState implements QualityAnalysisState<RealQualityAnalysisState> {

    @Getter
    private DoubleQuality realQuality;
    private RealQualityAddFunction addFunction;
    private RealQualityMergeFunction mergeFunction;

    public RealQualityAnalysisState(DoubleMetaData realMetaData) {
        this.realQuality = new DoubleQuality();
        this.addFunction = new RealQualityAddFunction(realMetaData);
        this.mergeFunction = new RealQualityMergeFunction();
    }

    public RealQualityAnalysisState add(Writable writable) {
        realQuality = addFunction.apply(realQuality, writable);
        return this;
    }

    public RealQualityAnalysisState merge(RealQualityAnalysisState other) {
        realQuality = mergeFunction.apply(realQuality, other.getRealQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return realQuality;
    }
}
