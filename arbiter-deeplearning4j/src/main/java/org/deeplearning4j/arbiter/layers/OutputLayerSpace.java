/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
 *
 */
package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.util.CollectionUtils;
import org.deeplearning4j.nn.conf.layers.OutputLayer;

/**
 * Layer hyperparameter configuration space for output layers
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class OutputLayerSpace extends BaseOutputLayerSpace<OutputLayer> {

    private OutputLayerSpace(Builder builder) {
        super(builder);

        this.numParameters = CollectionUtils.countUnique(collectLeaves());
    }

    @Override
    public OutputLayer getValue(double[] values) {
        OutputLayer.Builder o = new OutputLayer.Builder();
        setLayerOptionsBuilder(o, values);
        return o.build();
    }

    protected void setLayerOptionsBuilder(OutputLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
    }

    public static class Builder extends BaseOutputLayerSpace.Builder<Builder> {

        @Override
        @SuppressWarnings("unchecked")
        public OutputLayerSpace build() {
            return new OutputLayerSpace(this);
        }
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        return "OutputLayerSpace(" + super.toString(delim) + ")";
    }
}
