/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.nn.conf.override;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * Interface for a function to overrideLayer
 * builder configurations at a particular layer
 *
 */
@Deprecated
public interface ConfOverride {
    /**
     * Override the builder configuration for a particular layer
     * @param i the index of the layer
     * @param builder the layer builder to override values for
     */
    void overrideLayer(int i, NeuralNetConfiguration.Builder builder);

}
