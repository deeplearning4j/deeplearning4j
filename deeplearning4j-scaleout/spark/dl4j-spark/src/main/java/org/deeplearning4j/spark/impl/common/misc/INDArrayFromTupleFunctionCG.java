/*
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

package org.deeplearning4j.spark.impl.common.misc;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple3;

public class INDArrayFromTupleFunctionCG implements Function<Tuple3<INDArray,ComputationGraphUpdater,ScoreReport>,INDArray> {
    @Override
    public INDArray call(Tuple3<INDArray, ComputationGraphUpdater, ScoreReport> indArrayTuple2) throws Exception {
        return indArrayTuple2._1();
    }
}