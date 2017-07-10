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

package org.datavec.spark.transform.quality.string;

import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;
import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.quality.columns.StringQuality;
import org.datavec.api.writable.NullWritable;
import org.datavec.api.writable.Writable;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class StringQualityAddFunction implements Function2<StringQuality, Writable, StringQuality> {

    private final StringMetaData meta;

    @Override
    public StringQuality call(StringQuality v1, Writable writable) throws Exception {
        long valid = v1.getCountValid();
        long invalid = v1.getCountInvalid();
        long countMissing = v1.getCountMissing();
        long countTotal = v1.getCountTotal() + 1;
        long empty = v1.getCountEmptyString();
        long alphabetic = v1.getCountAlphabetic();
        long numerical = v1.getCountNumerical();
        long word = v1.getCountWordCharacter();
        long whitespaceOnly = v1.getCountWhitespace();
        HyperLogLogPlus hll = v1.getHll();

        String str = writable.toString();

        if (writable instanceof NullWritable)
            countMissing++;
        else if (meta.isValid(writable))
            valid++;
        else
            invalid++;

        if (str == null || str.isEmpty()) {
            empty++;
        } else {
            if (str.matches("[a-zA-Z]"))
                alphabetic++;
            if (str.matches("\\d+"))
                numerical++;
            if (str.matches("\\w+"))
                word++;
            if (str.matches("\\s+"))
                whitespaceOnly++;
        }

        hll.offer(str);
        return new StringQuality(valid, invalid, countMissing, countTotal, empty, alphabetic, numerical, word,
                        whitespaceOnly, hll);
    }
}
