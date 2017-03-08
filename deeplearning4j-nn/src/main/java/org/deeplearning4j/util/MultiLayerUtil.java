/*-
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

package org.deeplearning4j.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Various cooccurrences for manipulating a multi layer network
 * @author Adam Gibson
 */
public class MultiLayerUtil {
    private MultiLayerUtil() {}

    /**
     * Return the weight matrices for a multi layer network
     * @param network the network to get the weights for
     * @return the weight matrices for a given multi layer network
     */
    public static List<INDArray> weightMatrices(MultiLayerNetwork network) {
        List<INDArray> ret = new ArrayList<>();
        for (int i = 0; i < network.getLayers().length; i++) {
            ret.add(network.getLayers()[i].getParam(DefaultParamInitializer.WEIGHT_KEY));
        }
        return ret;
    }


}
