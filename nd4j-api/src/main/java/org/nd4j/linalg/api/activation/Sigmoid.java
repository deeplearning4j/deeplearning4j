/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.activation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactories;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;

/**
 * Sigmoid function (complex AND real!)
 * <p/>
 * http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1742-18.pdf
 * <p/>
 * For complex we set the k = 1
 *
 * @author Adam Gibson
 */
public class Sigmoid extends BaseActivationFunction {


    /**
     *
     */
    private static final long serialVersionUID = -6280602270833101092L;
    private int k = 1;


    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public ElementWiseOpFactory transformFactory() {
        return ElementWiseOpFactories.sigmoid();
    }

    @Override
    public INDArray applyDerivative(INDArray input) {
        INDArray rSub = input.rsub(1);
        return input.mul(rSub);
    }


}
