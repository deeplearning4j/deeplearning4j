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

package org.datavec.spark.transform.analysis.seqlength;

import org.apache.spark.api.java.function.Function2;

/**
 * Created by Alex on 7/03/2016.
 */
public class SequenceLengthAnalysisAddFunction
                implements Function2<SequenceLengthAnalysisCounter, Integer, SequenceLengthAnalysisCounter> {

    @Override
    public SequenceLengthAnalysisCounter call(SequenceLengthAnalysisCounter v1, Integer length) throws Exception {

        long zero = v1.getCountZeroLength();
        long one = v1.getCountOneLength();

        if (length == 0)
            zero++;
        else if (length == 1)
            one++;

        int newMinValue;
        long countMinValue = v1.getCountMinLength();
        if (length == v1.getMinLengthSeen()) {
            newMinValue = length;
            countMinValue++;
        } else if (v1.getMinLengthSeen() > length) {
            newMinValue = length;
            countMinValue = 1;
        } else {
            newMinValue = v1.getMinLengthSeen();
            //no change to count
        }

        int newMaxValue;
        long countMaxValue = v1.getCountMaxLength();
        if (length == v1.getMaxLengthSeen()) {
            newMaxValue = length;
            countMaxValue++;
        } else if (v1.getMaxLengthSeen() < length) {
            //reset max counter
            newMaxValue = length;
            countMaxValue = 1;
        } else {
            newMaxValue = v1.getMaxLengthSeen();
            //no change to count
        }

        //New mean:
        double sum = v1.getMean() * v1.getCountTotal() + length;
        long newTotalCount = v1.getCountTotal() + 1;
        double newMean = sum / newTotalCount;

        return new SequenceLengthAnalysisCounter(zero, one, countMinValue, newMinValue, countMaxValue, newMaxValue,
                        newTotalCount, newMean);
    }
}
