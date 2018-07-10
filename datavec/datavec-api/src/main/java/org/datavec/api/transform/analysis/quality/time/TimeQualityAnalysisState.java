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

package org.datavec.api.transform.analysis.quality.time;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.TimeMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.TimeQuality;
import org.datavec.api.writable.Writable;

/**
 * @author Alex Black
 */
public class TimeQualityAnalysisState implements QualityAnalysisState<TimeQualityAnalysisState> {

    @Getter
    private TimeQuality timeQuality;
    private TimeQualityAddFunction addFunction;
    private TimeQualityMergeFunction mergeFunction;

    public TimeQualityAnalysisState(TimeMetaData timeMetaData) {
        this.timeQuality = new TimeQuality();
        this.addFunction = new TimeQualityAddFunction(timeMetaData);
        this.mergeFunction = new TimeQualityMergeFunction();
    }

    public TimeQualityAnalysisState add(Writable writable) {
        timeQuality = addFunction.apply(timeQuality, writable);
        return this;
    }

    public TimeQualityAnalysisState merge(TimeQualityAnalysisState other) {
        timeQuality = mergeFunction.apply(timeQuality, other.getTimeQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return timeQuality;
    }
}
