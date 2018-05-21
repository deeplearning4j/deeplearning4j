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

package org.datavec.api.transform.transform.doubletransform;

import lombok.Data;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Normalize using (x-mean)/stdev.
 * Also known as a standard score, standardization etc.
 *
 * @author Alex Black
 */
@Data
public class StandardizeNormalizer extends BaseDoubleTransform {

    protected final double mean;
    protected final double stdev;

    public StandardizeNormalizer(@JsonProperty("columnName") String columnName, @JsonProperty("mean") double mean,
                    @JsonProperty("stdev") double stdev) {
        super(columnName);
        this.mean = mean;
        this.stdev = stdev;
    }


    @Override
    public Writable map(Writable writable) {
        double val = writable.toDouble();
        return new DoubleWritable((val - mean) / stdev);
    }

    @Override
    public String toString() {
        return "StandardizeNormalizer(mean=" + mean + ",stdev=" + stdev + ")";
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        Number n = (Number) input;
        double val = n.doubleValue();
        return new DoubleWritable((val - mean) / stdev);
    }
}
