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

package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer;

/**
 *  LSTM layer initializer.
 *  For LSTM based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 */
public class GravesBidirectionalLSTMLayerFactory extends DefaultLayerFactory {

    public GravesBidirectionalLSTMLayerFactory(Class<? extends Layer> layerConfig) {
        super(layerConfig);
    }

    @Override
    public ParamInitializer initializer() {
        return new GravesBidirectionalLSTMParamInitializer();
    }
}
