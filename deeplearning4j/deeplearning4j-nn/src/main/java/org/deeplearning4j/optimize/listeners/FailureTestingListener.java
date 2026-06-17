/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.optimize.listeners;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.*;

@Slf4j
public class FailureTestingListener implements TrainingListener, Serializable {

    private static final long serialVersionUID = 1L;

    private final FailureTrigger trigger;
    private final FailureMode failureMode;

    public FailureTestingListener(@NonNull FailureMode mode, @NonNull FailureTrigger trigger){
        this.trigger = trigger;
        this.failureMode = mode;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        call(CallType.ITER_DONE, model);
    }

    @Override
    public void onEpochStart(Model model) {
        call(CallType.EPOCH_START, model);
    }

    @Override
    public void onEpochEnd(Model model) {
        call(CallType.EPOCH_END, model);
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        call(CallType.FORWARD_PASS, model);
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        call(CallType.FORWARD_PASS, model);
    }

    @Override
    public void onGradientCalculation(Model model) {
        call(CallType.GRADIENT_CALC, model);
    }

    @Override
    public void onBackwardPass(Model model) {
        call(CallType.BACKWARD_PASS, model);
    }

    protected void call(CallType callType, Model model){
        if(!trigger.initialized()){
            trigger.initialize();
        }

        int iter;
        int epoch;
        if(model instanceof MultiLayerNetwork){
            iter = ((MultiLayerNetwork) model).getIterationCount();
            epoch = ((MultiLayerNetwork) model).getEpochCount();
        } else {
            iter = ((ComputationGraph) model).getIterationCount();
            epoch = ((ComputationGraph) model).getEpochCount();
        }
        boolean triggered = trigger.triggerFailure(callType, iter, epoch, model);

        if(triggered){
            log.error("*** FailureTestingListener was triggered on iteration {}, epoch {} - Failure mode is set to {} ***",
                    iter, epoch, failureMode);
            switch (failureMode){
                case OOM:
                    List<INDArray> list = new ArrayList<>();
                    while(true){
                        INDArray arr = Nd4j.createUninitialized(1_000_000_000);
                        list.add(arr);
                    }
                    //break;
                case SYSTEM_EXIT_1:
                    log.error("Exiting due to FailureTestingListener triggering - calling System.exit(1)");
                    System.exit(1);
                    break;
                case ILLEGAL_STATE:
                    log.error("Throwing new IllegalStateException due to FailureTestingListener triggering");
                    throw new IllegalStateException("FailureTestListener was triggered with failure mode " + failureMode
                    + " - iteration " + iter + ", epoch " + epoch);
                case INFINITE_SLEEP:
                    while(true){
                        try {
                            Thread.sleep(10000);
                        } catch (InterruptedException e){
                            //Ignore
                        }
                    }
                default:
                    throw new RuntimeException("Unknown enum value: " + failureMode);
            }
        }
    }


}
