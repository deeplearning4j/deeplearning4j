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

package org.deeplearning4j.optimize.listeners;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.net.InetAddress;
import java.util.*;

/**
 * WARNING: THIS LISTENER SHOULD ONLY BE USED FOR MANUAL TESTING PURPOSES<br>
 * It intentionally causes various types of failures according to some criteria, in order to test the response
 * to it.<br>
 * This is useful for example in:
 * (a) Testing Spark fault tolerance<br>
 * (b) Testing OOM exception crash dump information<br>
 * Generally it should not be used in unit tests either, depending on how it is configured.<br>
 * <br>
 * Two aspects need to be configured to use this listener:
 * 1. If/when the "failure" should be triggered - via FailureTrigger classes<br>
 * 2. The type of failure when triggered - via FailureMode enum<br>
 * <br>
 * To specify if/when a failure should be triggered, use a {@link FailureTrigger} instance. Some built-in ones
 * are provided, random probability, time since initialized, username, and iteration/epoch count.
 * <br>
 * Types of failures available:<br>
 * - OOM (allocate large arrays in loop until OOM).<br>
 * - System.exit(1)<br>
 * - IllegalStateException<br>
 * - Infinite sleep<br>
 *
 * @author Alex Black
 */
@Slf4j
public class FailureTestingListener implements TrainingListener, Serializable {

    public enum FailureMode {OOM, SYSTEM_EXIT_1, ILLEGAL_STATE, INFINITE_SLEEP}
    public enum CallType {ANY, EPOCH_START, EPOCH_END, FORWARD_PASS, GRADIENT_CALC, BACKWARD_PASS, ITER_DONE}

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


    @Data
    public static abstract class FailureTrigger implements Serializable {

        private boolean initialized = false;

        /**
         * If true: trigger the failure. If false: don't trigger failure
         * @param callType  Type of call
         * @param iteration Iteration number
         * @param epoch     Epoch number
         * @param model     Model
         * @return
         */
        public abstract boolean triggerFailure(CallType callType, int iteration, int epoch, Model model);

        public boolean initialized(){
            return initialized;
        }

        public void initialize(){
            this.initialized = true;
        }
    }

    @AllArgsConstructor
    public static class And extends FailureTrigger{

        protected List<FailureTrigger> triggers;

        public And(FailureTrigger... triggers){
            this.triggers = Arrays.asList(triggers);
        }

        @Override
        public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
            boolean b = true;
            for(FailureTrigger ft : triggers)
                b &= ft.triggerFailure(callType, iteration, epoch, model);
            return b;
        }

        @Override
        public void initialize(){
            super.initialize();
            for(FailureTrigger ft : triggers)
                ft.initialize();
        }
    }

    public static class Or extends And {
        public Or(FailureTrigger... triggers) {
            super(triggers);
        }

        @Override
        public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
            boolean b = false;
            for(FailureTrigger ft : triggers)
                b |= ft.triggerFailure(callType, iteration, epoch, model);
            return b;
        }
    }

    @Data
    public static class RandomProb extends FailureTrigger {

        private final CallType callType;
        private final double probability;
        private Random rng;

        public RandomProb(CallType callType, double probability){
            this.callType = callType;
            this.probability = probability;
        }

        @Override
        public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
            return (this.callType == CallType.ANY || callType == this.callType) && rng.nextDouble() < probability;
        }

        @Override
        public void initialize(){
            super.initialize();
            this.rng = new Random();
        }
    }


    @Data
    public static class TimeSinceInitializedTrigger extends FailureTrigger {

        private final long msSinceInit;
        private long initTime;

        public TimeSinceInitializedTrigger(long msSinceInit){
            this.msSinceInit = msSinceInit;
        }

        @Override
        public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
            return (System.currentTimeMillis() - initTime) > msSinceInit;
        }

        @Override
        public void initialize(){
            super.initialize();
            this.initTime = System.currentTimeMillis();
        }
    }

    @Data
    public static class UserNameTrigger extends FailureTrigger {
        private final String userName;
        private boolean shouldFail = false;

        public UserNameTrigger(@NonNull String userName) {
            this.userName = userName;
        }


        @Override
        public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
            return shouldFail;
        }

        @Override
        public void initialize(){
            super.initialize();
            shouldFail = this.userName.equalsIgnoreCase(System.getProperty("user.name"));
        }
    }
    //System.out.println("Hostname: " + InetAddress.getLocalHost().getHostName());

    @Data
    public static class HostNameTrigger extends FailureTrigger{
        private final String hostName;
        private boolean shouldFail = false;

        public HostNameTrigger(@NonNull String hostName) {
            this.hostName = hostName;
        }


        @Override
        public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
            return shouldFail;
        }

        @Override
        public void initialize(){
            super.initialize();
            try {
                String hostname = InetAddress.getLocalHost().getHostName();
                log.info("FailureTestingListere hostname: {}", hostname);
                shouldFail = this.hostName.equalsIgnoreCase(hostname);
            } catch (Exception e){
                throw new RuntimeException(e);
            }
        }
    }

    @Data
    public static class IterationEpochTrigger extends FailureTrigger {

        private final boolean isEpoch;
        private final int count;

        public IterationEpochTrigger(boolean isEpoch, int count){
            this.isEpoch = isEpoch;
            this.count = count;
        }

        @Override
        public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
            return (isEpoch && epoch == count) || (!isEpoch && iteration == count);
        }
    }


}
