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

package org.datavec.api.transform.transform.doubletransform;

import lombok.Data;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Normalize by substracting the mean
 */
@Data
public class SubtractMeanNormalizer extends BaseDoubleTransform {

    private final double mean;

    public SubtractMeanNormalizer(@JsonProperty("columnName") String columnName, @JsonProperty("mean") double mean) {
        super(columnName);
        this.mean = mean;
    }

    @Override
    public Writable map(Writable writable) {
        return new DoubleWritable(writable.toDouble() - mean);
    }

    @Override
    public String toString() {
        return "SubstractMeanNormalizer(mean=" + mean + ")";
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
        return n.doubleValue() - mean;
    }
}
