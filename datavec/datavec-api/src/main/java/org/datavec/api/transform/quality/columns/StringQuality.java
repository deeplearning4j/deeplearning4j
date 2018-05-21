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

package org.datavec.api.transform.quality.columns;

import com.clearspring.analytics.stream.cardinality.CardinalityMergeException;
import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Created by Alex on 5/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class StringQuality extends ColumnQuality {

    private final long countEmptyString; //"" string
    private final long countAlphabetic; //A-Z, a-z only
    private final long countNumerical; //0-9 only
    private final long countWordCharacter; //A-Z, a-z, 0-9
    private final long countWhitespace; //tab, spaces etc ONLY
    private final HyperLogLogPlus hll;

    public StringQuality() {
        this(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05);
    }

    public StringQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countEmptyString,
                    long countAlphabetic, long countNumerical, long countWordCharacter, long countWhitespace,
                    HyperLogLogPlus hll) {
        super(countValid, countInvalid, countMissing, countTotal);
        this.countEmptyString = countEmptyString;
        this.countAlphabetic = countAlphabetic;
        this.countNumerical = countNumerical;
        this.countWordCharacter = countWordCharacter;
        this.countWhitespace = countWhitespace;
        this.hll = hll;
    }

    public StringQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countEmptyString,
                    long countAlphabetic, long countNumerical, long countWordCharacter, long countWhitespace,
                    double relativeSD) {
        /*
         * The algorithm used is based on streamlib's implementation of "HyperLogLog in Practice:
         * Algorithmic Engineering of a State of The Art Cardinality Estimation Algorithm", available
         * <a href="http://dx.doi.org/10.1145/2452376.2452456">here</a>.
         *
         * The relative accuracy is approximately `1.054 / sqrt(2^p)`. Setting
         * a nonzero `sp > p` in HyperLogLogPlus(p, sp) would trigger sparse
         * representation of registers, which may reduce the memory consumption
         * and increase accuracy when the cardinality is small.
         */
        this(countValid, countInvalid, countMissing, countTotal, countEmptyString, countAlphabetic, countNumerical,
                        countWordCharacter, countWhitespace,
                        new HyperLogLogPlus((int) Math.ceil(2.0 * Math.log(1.054 / relativeSD) / Math.log(2)), 0));
    }

    public StringQuality add(StringQuality other) throws CardinalityMergeException {
        hll.addAll(other.hll);
        return new StringQuality(countValid + other.countValid, countInvalid + other.countInvalid,
                        countMissing + other.countMissing, countTotal + other.countTotal,
                        countEmptyString + other.countEmptyString, countAlphabetic + other.countAlphabetic,
                        countNumerical + other.countNumerical, countWordCharacter + other.countWordCharacter,
                        countWhitespace + other.countWhitespace, hll);
    }

    @Override
    public String toString() {
        return "StringQuality(" + super.toString() + ", countEmptyString=" + countEmptyString + ", countAlphabetic="
                        + countAlphabetic + ", countNumerical=" + countNumerical + ", countWordCharacter="
                        + countWordCharacter + ", countWhitespace=" + countWhitespace + ", countApproxUnique="
                        + hll.cardinality() + ")";
    }

}
