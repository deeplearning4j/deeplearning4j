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

import lombok.AllArgsConstructor;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.quality.columns.DoubleQuality;
import org.datavec.api.writable.NullWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.BiFunction;

import java.io.Serializable;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class RealQualityAddFunction implements BiFunction<DoubleQuality, Writable, DoubleQuality>, Serializable {

    private final DoubleMetaData meta;

    @Override
    public DoubleQuality apply(DoubleQuality v1, Writable writable) {

        long valid = v1.getCountValid();
        long invalid = v1.getCountInvalid();
        long countMissing = v1.getCountMissing();
        long countTotal = v1.getCountTotal() + 1;
        long nonReal = v1.getCountNonReal();
        long nan = v1.getCountNaN();
        long infinite = v1.getCountInfinite();

        if (meta.isValid(writable))
            valid++;
        else if (writable instanceof NullWritable
                        || writable instanceof Text && (writable.toString() == null || writable.toString().isEmpty()))
            countMissing++;
        else
            invalid++;

        String str = writable.toString();
        double d;
        try {
            d = Double.parseDouble(str);
            if (Double.isNaN(d))
                nan++;
            if (Double.isInfinite(d))
                infinite++;
        } catch (NumberFormatException e) {
            nonReal++;
        }

        return new DoubleQuality(valid, invalid, countMissing, countTotal, nonReal, nan, infinite);
    }
}
