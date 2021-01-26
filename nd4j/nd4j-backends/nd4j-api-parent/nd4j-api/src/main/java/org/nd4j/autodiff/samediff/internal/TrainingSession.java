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

package org.nd4j.autodiff.samediff.internal;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.common.primitives.Pair;

import java.util.*;

/**
 * TrainingSession extends InferenceSession, to add training-specific functionality:<br>
 * - Application of regularization (L1, L2, weight decay etc)<br>
 * - Inline updating of variables, using updater/optimizer (Adam, Nesterov, SGD, etc)<br>
 * - Calculation of regularization scores (Score for L1, L2, etc)
 *
 * @author Alex Black
 */
@Slf4j
public class TrainingSession extends InferenceSession {

    protected TrainingConfig config;
    protected Map<String, String> gradVarToVarMap;
    protected Map<String, GradientUpdater> updaters;
    protected Map<String, Integer> lossVarsToLossIdx;
    protected double[] currIterLoss;
    protected Map<Class<?>, AtomicDouble> currIterRegLoss;
    protected List<Listener> listeners;


    public TrainingSession(SameDiff sameDiff) {
        super(sameDiff);
    }

    /**
     * Perform one iteration of training - i.e., do forward and backward passes, and update the parameters
     *
     * @param config        Training configuration
     * @param placeholders  Current placeholders
     * @param paramsToTrain Set of parameters that will be trained
     * @param updaters      Current updater state
     * @param batch         Current data/batch (mainly for listeners, should have already been converted to placeholders map)
     * @param lossVariables Loss variables (names)
     * @param listeners     Listeners (if any)
     * @param at            Current epoch, iteration, etc
     * @return The Loss at the current iteration
     */
    public Loss trainingIteration(TrainingConfig config, Map<String, INDArray> placeholders, Set<String> paramsToTrain, Map<String, GradientUpdater> updaters,
                                  MultiDataSet batch, List<String> lossVariables, List<Listener> listeners, At at) {
        this.config = config;
        this.updaters = updaters;

        //Preprocess listeners, get the relevant ones
        if (listeners == null) {
            this.listeners = null;
        } else {
            List<Listener> filtered = new ArrayList<>();
            for (Listener l : listeners) {
                if (l.isActive(at.operation())) {
                    filtered.add(l);
                }
            }
            this.listeners = filtered.isEmpty() ? null : filtered;
        }

        Set<String> requiredActivations = new HashSet<>();
        gradVarToVarMap = new HashMap<>();       //Key: gradient variable. Value: variable that the key is gradient for
        for (String s : paramsToTrain) {
            Preconditions.checkState(sameDiff.hasVariable(s), "SameDiff instance does not have a variable with name \"%s\"", s);
            SDVariable v = sameDiff.getVariable(s);
            Preconditions.checkState(v.getVariableType() == VariableType.VARIABLE, "Can only train VARIABLE type variable - \"%s\" has type %s",
                    s, v.getVariableType());
            SDVariable grad = sameDiff.getVariable(s).getGradient();
            if (grad == null) {
                //In some cases, a variable won't actually impact the loss value, and hence won't have a gradient associated with it
                //For example: floatVar -> cast to integer -> cast to float -> sum -> loss
                //In this case, the gradient of floatVar isn't defined (due to no floating point connection to the loss)
                continue;
            }

            requiredActivations.add(grad.name());

            gradVarToVarMap.put(grad.name(), s);
        }

        //Also add evaluations - in case we want to evaluate something that isn't required to determine loss
        // (hence wouldn't normally be calculated)
        if(config.getTrainEvaluations() != null){
            requiredActivations.addAll(config.getTrainEvaluations().keySet());
        }

        //Set up losses
        lossVarsToLossIdx = new LinkedHashMap<>();
        List<String> lossVars;
        currIterLoss = new double[lossVariables.size()];
        currIterRegLoss = new HashMap<>();
        for (int i = 0; i < lossVariables.size(); i++) {
            lossVarsToLossIdx.put(lossVariables.get(i), i);
        }

        //Do training iteration
        List<String> outputVars = new ArrayList<>(gradVarToVarMap.keySet());    //TODO this should be empty, and grads calculated in requiredActivations
        Map<String, INDArray> m = output(outputVars, placeholders, batch, requiredActivations, listeners, at);


        double[] finalLoss = new double[currIterLoss.length + currIterRegLoss.size()];
        System.arraycopy(currIterLoss, 0, finalLoss, 0, currIterLoss.length);
        if (currIterRegLoss.size() > 0) {
            lossVars = new ArrayList<>(lossVariables.size() + currIterRegLoss.size());
            lossVars.addAll(lossVariables);
            int s = currIterRegLoss.size();
            //Collect regularization losses
            for (Map.Entry<Class<?>, AtomicDouble> entry : currIterRegLoss.entrySet()) {
                lossVars.add(entry.getKey().getSimpleName());
                finalLoss[s] = entry.getValue().get();
            }
        } else {
            lossVars = lossVariables;
        }

        Loss loss = new Loss(lossVars, finalLoss);
        if (listeners != null) {
            for (Listener l : listeners) {
                if (l.isActive(Operation.TRAINING)) {
                    l.iterationDone(sameDiff, at, batch, loss);
                }
            }
        }

        return loss;
    }

    @Override
    public INDArray[] getOutputs(Pair<SameDiffOp, OpContext> opPair, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                 Set<String> constAndPhInputs, List<Listener> listeners, At at, MultiDataSet batch, Set<String> allReqVariables) {
        //Get outputs from InferenceSession
        INDArray[] out = super.getOutputs(opPair, outputFrameIter, opInputs, allIterInputs, constAndPhInputs, listeners, at, batch, allReqVariables);
        SameDiffOp op = opPair.getFirst();

        List<String> outputs = op.getOutputsOfOp();
        int outIdx = 0;
        for (String s : outputs) {
            //If this is a loss variable - record it
            if (lossVarsToLossIdx.containsKey(s)) {
                int lossIdx = lossVarsToLossIdx.get(s);
                INDArray arr = out[outIdx];
                double l = arr.isScalar() ? arr.getDouble(0) : arr.sumNumber().doubleValue();
                currIterLoss[lossIdx] += l;
            }

            //If this is a gradient variable - apply the updater and update the parameter array in-line
            if (gradVarToVarMap.containsKey(s)) {
                String varName = gradVarToVarMap.get(s);
                //log.info("Calculated gradient for variable \"{}\": (grad var name: \"{}\")", varName, s);

                Variable gradVar = sameDiff.getVariables().get(s);
                if (gradVar.getInputsForOp() != null && gradVar.getInputsForOp().isEmpty()) {
                    //Should be rare, and we should handle this by tracking dependencies, and only update when safe
                    // (i.e., dependency tracking)
                    throw new IllegalStateException("Op depends on gradient variable: " + s + " for variable " + varName);
                }

                GradientUpdater u = updaters.get(varName);
                Preconditions.checkState(u != null, "No updater found for variable \"%s\"", varName);

                Variable var = sameDiff.getVariables().get(varName);
                INDArray gradArr = out[outIdx];
                INDArray paramArr = var.getVariable().getArr();

                //Pre-updater regularization (L1, L2)
                List<Regularization> r = config.getRegularization();
                if (r != null && r.size() > 0) {
                    double lr = config.getUpdater().hasLearningRate() ? config.getUpdater().getLearningRate(at.iteration(), at.epoch()) : 1.0;
                    for (Regularization reg : r) {
                        if (reg.applyStep() == Regularization.ApplyStep.BEFORE_UPDATER) {
                            if (this.listeners != null) {
                                double score = reg.score(paramArr, at.iteration(), at.epoch());
                                if (!currIterRegLoss.containsKey(reg.getClass())) {
                                    currIterRegLoss.put(reg.getClass(), new AtomicDouble());
                                }
                                currIterRegLoss.get(reg.getClass()).addAndGet(score);
                            }
                            reg.apply(paramArr, gradArr, lr, at.iteration(), at.epoch());
                        }
                    }
                }

                u.applyUpdater(gradArr, at.iteration(), at.epoch());

                //Post-apply regularization (weight decay)
                if (r != null && r.size() > 0) {
                    double lr = config.getUpdater().hasLearningRate() ? config.getUpdater().getLearningRate(at.iteration(), at.epoch()) : 1.0;
                    for (Regularization reg : r) {
                        if (reg.applyStep() == Regularization.ApplyStep.POST_UPDATER) {
                            if (this.listeners != null) {
                                double score = reg.score(paramArr, at.iteration(), at.epoch());
                                if (!currIterRegLoss.containsKey(reg.getClass())) {
                                    currIterRegLoss.put(reg.getClass(), new AtomicDouble());
                                }
                                currIterRegLoss.get(reg.getClass()).addAndGet(score);
                            }
                            reg.apply(paramArr, gradArr, lr, at.iteration(), at.epoch());
                        }
                    }
                }

                if (listeners != null) {
                    for (Listener l : listeners) {
                        if (l.isActive(at.operation()))
                            l.preUpdate(sameDiff, at, var, gradArr);
                    }
                }

                //Update:
                if (config.isMinimize()) {
                    paramArr.subi(gradArr);
                } else {
                    paramArr.addi(gradArr);
                }
                log.trace("Applied updater to gradient and updated variable: {}", varName);
            }

            outIdx++;
        }

        return out;
    }
}
