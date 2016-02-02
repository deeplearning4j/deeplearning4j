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

package org.deeplearning4j.earlystopping.termination;

/** Iteration termination condition for terminating training if the minibatch score exceeds a certain value.
 * This can occur for example with a poorly tuned (too high) learning rate
 */
public class MaxScoreIterationTerminationCondition implements IterationTerminationCondition {

    private double maxScore;

    public MaxScoreIterationTerminationCondition(double maxScore) {
        this.maxScore = maxScore;
    }

    @Override
    public void initialize() {
        //no op
    }

    @Override
    public boolean terminate(double lastMiniBatchScore) {
        return lastMiniBatchScore > maxScore;
    }

    @Override
    public String toString(){
        return "MaxScoreIterationTerminationCondition("+maxScore+")";
    }
}
