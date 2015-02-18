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

package org.nd4j.linalg.distancefunction;


import org.nd4j.linalg.api.ndarray.INDArray;

public class ManhattanDistance extends BaseDistanceFunction {

    /**
     *
     */
    private static final long serialVersionUID = -2421779223755051432L;

    public ManhattanDistance(INDArray base) {
        super(base);
    }

    @Override
    public Float apply(INDArray input) {
        return (float) base.distance1(input);
    }


}
