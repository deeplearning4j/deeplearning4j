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
    private final long countUnique;

    public StringQuality() {
        this(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    public StringQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countEmptyString,
                    long countAlphabetic, long countNumerical, long countWordCharacter, long countWhitespace,
                    long countUnique) {
        super(countValid, countInvalid, countMissing, countTotal);
        this.countEmptyString = countEmptyString;
        this.countAlphabetic = countAlphabetic;
        this.countNumerical = countNumerical;
        this.countWordCharacter = countWordCharacter;
        this.countWhitespace = countWhitespace;
        this.countUnique = countUnique;
    }


    public StringQuality add(StringQuality other) {
        return new StringQuality(countValid + other.countValid, countInvalid + other.countInvalid,
                        countMissing + other.countMissing, countTotal + other.countTotal,
                        countEmptyString + other.countEmptyString, countAlphabetic + other.countAlphabetic,
                        countNumerical + other.countNumerical, countWordCharacter + other.countWordCharacter,
                        countWhitespace + other.countWhitespace, countUnique + other.countUnique);
    }

    @Override
    public String toString() {
        return "StringQuality(" + super.toString() + ", countEmptyString=" + countEmptyString + ", countAlphabetic="
                        + countAlphabetic + ", countNumerical=" + countNumerical + ", countWordCharacter="
                        + countWordCharacter + ", countWhitespace=" + countWhitespace + ", countUnique=" + countUnique
                        + ")";
    }

}
