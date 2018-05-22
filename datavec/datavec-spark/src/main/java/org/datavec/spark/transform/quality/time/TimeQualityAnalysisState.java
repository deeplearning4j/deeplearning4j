/*
 *  * Copyright 2018 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.spark.transform.quality.time;

import lombok.Getter;
import org.datavec.api.transform.metadata.TimeMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.TimeQuality;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.quality.QualityAnalysisState;

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

    public TimeQualityAnalysisState add(Writable writable) throws Exception {
        timeQuality = addFunction.call(timeQuality, writable);
        return this;
    }

    public TimeQualityAnalysisState merge(TimeQualityAnalysisState other) throws Exception {
        timeQuality = mergeFunction.call(timeQuality, other.getTimeQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return timeQuality;
    }
}
