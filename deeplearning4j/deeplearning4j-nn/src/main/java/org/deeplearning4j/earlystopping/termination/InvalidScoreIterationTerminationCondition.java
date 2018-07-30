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

/** Terminate training at this iteration if score is NaN or Infinite for the last minibatch */
public class InvalidScoreIterationTerminationCondition implements IterationTerminationCondition {
    @Override
    public void initialize() {
        //No op
    }

    @Override
    public boolean terminate(double lastMiniBatchScore) {
        return Double.isNaN(lastMiniBatchScore) || Double.isInfinite(lastMiniBatchScore);
    }

    @Override
    public String toString() {
        return "InvalidScoreIterationTerminationCondition()";
    }
}
