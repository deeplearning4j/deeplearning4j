/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.network;

import lombok.Getter;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A {@link INetworkHandler} implementation to be used when multiple separate network are to be used as one. For example,
 * we can have two separate networks, <i>value</i> and <i>policy</i>, and use a CompoundNetworkHandler to use them the
 * same way as if it was a single combined network.
 *
 * Note: each individual network should have only one output layer.
 */
public class CompoundNetworkHandler implements INetworkHandler {

    private final INetworkHandler[] networkHandlers;
    @Getter
    private boolean recurrent;

    /**
     * @param networkHandlers All networks to be used in this instance.
     */
    public CompoundNetworkHandler(INetworkHandler... networkHandlers) {
        this.networkHandlers = networkHandlers;

        for(INetworkHandler handler : networkHandlers) {
            recurrent |= handler.isRecurrent();
        }
    }

    @Override
    public void notifyGradientCalculation() {
        for(INetworkHandler handler : networkHandlers) {
            handler.notifyGradientCalculation();
        }
    }

    @Override
    public void notifyIterationDone() {
        for(INetworkHandler handler : networkHandlers) {
            handler.notifyIterationDone();
        }
    }

    @Override
    public void performFit(FeaturesLabels featuresLabels) {
        for(INetworkHandler handler : networkHandlers) {
            handler.performFit(featuresLabels);
        }
    }

    @Override
    public void performGradientsComputation(FeaturesLabels featuresLabels) {
        for(INetworkHandler handler : networkHandlers) {
            handler.performGradientsComputation(featuresLabels);
        }
    }

    @Override
    public void fillGradientsResponse(Gradients gradients) {
        for(INetworkHandler handler : networkHandlers) {
            handler.fillGradientsResponse(gradients);
        }
    }

    @Override
    public void applyGradient(Gradients gradients, long batchSize) {
        for(INetworkHandler handler : networkHandlers) {
            handler.applyGradient(gradients, batchSize);
        }
    }

    @Override
    public INDArray[] recurrentStepOutput(Observation observation) {
        List<INDArray> outputs = new ArrayList<INDArray>();
        for(INetworkHandler handler : networkHandlers) {
            Collections.addAll(outputs, handler.recurrentStepOutput(observation));
        }

        return outputs.toArray(new INDArray[0]);
    }

    @Override
    public INDArray[] stepOutput(Observation observation) {
        List<INDArray> outputs = new ArrayList<INDArray>();
        for(INetworkHandler handler : networkHandlers) {
            Collections.addAll(outputs, handler.stepOutput(observation));
        }

        return outputs.toArray(new INDArray[0]);
    }

    @Override
    public INDArray[] batchOutput(Features features) {
        List<INDArray> outputs = new ArrayList<INDArray>();
        for(INetworkHandler handler : networkHandlers) {
            Collections.addAll(outputs, handler.batchOutput(features));
        }

        return outputs.toArray(new INDArray[0]);
    }

    @Override
    public void resetState() {
        for(INetworkHandler handler : networkHandlers) {
            if(handler.isRecurrent()) {
                handler.resetState();
            }
        }
    }

    @Override
    public INetworkHandler clone() {
        INetworkHandler[] clonedHandlers = new INetworkHandler[networkHandlers.length];
        for(int i = 0; i < networkHandlers.length; ++i) {
            clonedHandlers[i] = networkHandlers[i].clone();
        }

        return new CompoundNetworkHandler(clonedHandlers);
    }

    @Override
    public void copyFrom(INetworkHandler from) {
        for(int i = 0; i < networkHandlers.length; ++i) {
            networkHandlers[i].copyFrom(((CompoundNetworkHandler) from).networkHandlers[i]);
        }
    }
}
