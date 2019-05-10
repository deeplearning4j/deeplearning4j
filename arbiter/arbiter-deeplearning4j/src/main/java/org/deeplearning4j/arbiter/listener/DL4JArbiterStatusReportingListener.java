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

package org.deeplearning4j.arbiter.listener;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.List;

/**
 * A simple DL4J Iteration listener that calls Arbiter's status listeners
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class DL4JArbiterStatusReportingListener extends BaseTrainingListener {

    private List<StatusListener> statusListeners;
    private CandidateInfo candidateInfo;

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (statusListeners == null) {
            return;
        }

        for (StatusListener sl : statusListeners) {
            sl.onCandidateIteration(candidateInfo, model, iteration);
        }
    }
}
