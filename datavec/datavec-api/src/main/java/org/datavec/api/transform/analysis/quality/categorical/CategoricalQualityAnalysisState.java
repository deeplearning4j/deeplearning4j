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

package org.datavec.api.transform.analysis.quality.categorical;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.quality.columns.CategoricalQuality;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 3/6/17.
 */
public class CategoricalQualityAnalysisState implements QualityAnalysisState<CategoricalQualityAnalysisState> {

    @Getter
    private CategoricalQuality categoricalQuality;
    private CategoricalQualityAddFunction addFunction;
    private CategoricalQualityMergeFunction mergeFunction;

    public CategoricalQualityAnalysisState(CategoricalMetaData integerMetaData) {
        this.categoricalQuality = new CategoricalQuality();
        this.addFunction = new CategoricalQualityAddFunction(integerMetaData);
        this.mergeFunction = new CategoricalQualityMergeFunction();
    }

    public CategoricalQualityAnalysisState add(Writable writable) {
        categoricalQuality = addFunction.apply(categoricalQuality, writable);
        return this;
    }

    public CategoricalQualityAnalysisState merge(CategoricalQualityAnalysisState other) {
        categoricalQuality = mergeFunction.apply(categoricalQuality, other.getCategoricalQuality());
        return this;
    }


    public ColumnQuality getColumnQuality() {
        return categoricalQuality;
    }
}
