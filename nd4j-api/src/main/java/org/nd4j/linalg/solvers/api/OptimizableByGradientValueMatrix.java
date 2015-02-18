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

package org.nd4j.linalg.solvers.api;


import org.nd4j.linalg.api.ndarray.INDArray;

public interface OptimizableByGradientValueMatrix {

    public int getNumParameters();

    public INDArray getParameters();

    public void setParameters(INDArray params);

    public double getParameter(int index);

    public void setParameter(int index, double value);

    public INDArray getValueGradient(int iteration);


    public double getValue();


    void setCurrentIteration(int value);
}
