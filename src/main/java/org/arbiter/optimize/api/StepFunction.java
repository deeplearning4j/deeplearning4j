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

package org.arbiter.optimize.api;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Custom step function for line search
 *
 * @author Adam Gibson
 */
public interface StepFunction extends Serializable {

    /**
     * Step with the given parameters
     * @param x the current parameters
     * @param line the line to step
     * @param params
     */
    void step(INDArray x,INDArray line,Object[] params);


    /**
     * Step with no parameters
     */
    void step(INDArray x,INDArray line);


    void step();


}
