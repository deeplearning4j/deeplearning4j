/*-
 *  * Copyright 2016 Skymind, Inc.
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

package org.datavec.api.transform.analysis.counter;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.transform.analysis.AnalysisCounter;
import org.datavec.api.writable.Writable;

/**
 * A counter function for doing analysis on BytesWritable columns, on Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class BytesAnalysisCounter implements AnalysisCounter<BytesAnalysisCounter> {
    private long countTotal = 0;



    public BytesAnalysisCounter() {

    }


    @Override
    public BytesAnalysisCounter add(Writable writable) {
        countTotal++;

        return this;
    }

    public BytesAnalysisCounter merge(BytesAnalysisCounter other) {

        return new BytesAnalysisCounter(countTotal + other.countTotal);
    }

}
