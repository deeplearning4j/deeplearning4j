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

package org.deeplearning4j.optimize.api;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * A no-op implementation of a {@link TrainingListener} to be used as a starting point for custom training callbacks.
 *
 * Extend this and selectively override the methods you will actually use.
 */
public abstract class BaseTrainingListener implements TrainingListener {

    @Override
    public void onEpochStart(Model model) {
        //No op
    }


    @Override
    public void onEpochEnd(Model model) {
        //No op
    }


    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        //No op
    }


    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        //No op
    }


    @Override
    public void onGradientCalculation(Model model) {
        //No op
    }


    @Override
    public void onBackwardPass(Model model) {
        //No op
    }


    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        //No op
    }
}
