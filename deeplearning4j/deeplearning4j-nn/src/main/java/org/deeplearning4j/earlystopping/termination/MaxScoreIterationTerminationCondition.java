/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
        return lastMiniBatchScore > maxScore || Double.isNaN(lastMiniBatchScore);
    }

    @Override
    public String toString() {
        return "MaxScoreIterationTerminationCondition(" + maxScore + ")";
    }
}
