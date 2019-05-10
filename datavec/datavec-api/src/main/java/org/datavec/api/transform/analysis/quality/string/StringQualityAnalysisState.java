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

package org.datavec.api.transform.analysis.quality.string;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.StringQuality;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 3/6/17.
 */
public class StringQualityAnalysisState implements QualityAnalysisState<StringQualityAnalysisState> {

    @Getter
    private StringQuality stringQuality;
    private StringQualityAddFunction addFunction;
    private StringQualityMergeFunction mergeFunction;

    public StringQualityAnalysisState(StringMetaData stringMetaData) {
        this.stringQuality = new StringQuality();
        this.addFunction = new StringQualityAddFunction(stringMetaData);
        this.mergeFunction = new StringQualityMergeFunction();
    }

    public StringQualityAnalysisState add(Writable writable) {
        stringQuality = addFunction.apply(stringQuality, writable);
        return this;
    }

    public StringQualityAnalysisState merge(StringQualityAnalysisState other) {
        stringQuality = mergeFunction.apply(stringQuality, other.getStringQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return stringQuality;
    }
}
