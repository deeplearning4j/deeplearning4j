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

import java.io.Serializable;

/**
 * Training evaluator, used for determining early stop
 *
 * @author Adam Gibson
 */
public interface TrainingEvaluator extends Serializable {

    /**
     * Whether to terminate or  not
     *
     * @param epoch the current epoch
     * @return whether to terminate or not
     * on the given epoch
     */
    boolean shouldStop(int epoch);

    public double improvementThreshold();


    double patience();


    /**
     * Amount patience should be increased when a new best threshold is hit
     *
     * @return
     */
    double patienceIncrease();


    /**
     * The best validation loss so far
     *
     * @return the best validation loss so far
     */
    public double bestLoss();

    /**
     * The number of epochs to test on
     *
     * @return the number of epochs to test on
     */
    public int validationEpochs();


    public int miniBatchSize();

}
