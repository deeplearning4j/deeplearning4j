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

package org.nd4j.autodiff.listeners;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.nd4j.autodiff.listeners.records.EvaluationRecord;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * A base listener class that will preform the provided evaluations, and provide the results in epochEnd and validationDone
 *
 * Instead of overriding requiredVariables, epochStart, epochEnd, validationDone, and/or opExecution,
 * override otherRequiredVariables, epochStartEvaluations, epochEndEvaluations, validationDoneEvaluations, and/or opExecutionEvaluations
 *
 * <strong>If you want to use Evaluations in your listener, extend this class</strong>
 */
public abstract class BaseEvaluationListener extends BaseListener {

    private Map<String, List<IEvaluation>> trainingEvaluations = new HashMap<>();
    private Map<String, List<IEvaluation>> validationEvaluations = new HashMap<>();

    /**
     * Return the requested evaluations.  New instances of these evaluations will be made each time they are used
     */
    public abstract ListenerEvaluations evaluations();

    @Override
    public final ListenerVariables requiredVariables(SameDiff sd) {
        return evaluations().requiredVariables().merge(otherRequiredVariables(sd));
    }

    /**
     * Return any requested variables that are not part of the evaluations
     */
    public ListenerVariables otherRequiredVariables(SameDiff sd){
        return ListenerVariables.empty();
    }


    @Override
    public final void epochStart(SameDiff sd, At at) {
        trainingEvaluations = new HashMap<>();
        for(Map.Entry<String, List<IEvaluation>> entry : evaluations().trainEvaluations().entrySet()){

            List<IEvaluation> evals = new ArrayList<>();
            for(IEvaluation ie : entry.getValue())
                evals.add(ie.newInstance());

            trainingEvaluations.put(entry.getKey(), evals);
        }
        validationEvaluations = new HashMap<>();
        for(Map.Entry<String, List<IEvaluation>> entry : evaluations().validationEvaluations().entrySet()){

            List<IEvaluation> evals = new ArrayList<>();
            for(IEvaluation ie : entry.getValue())
                evals.add(ie.newInstance());

            validationEvaluations.put(entry.getKey(), evals);
        }

        epochStartEvaluations(sd, at);
    }

    /**
     * See {@link Listener#epochStart(SameDiff, At)}
     */
    public void epochStartEvaluations(SameDiff sd, At at){
        //No op
    }

    @Override
    public final ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis) {
        return epochEndEvaluations(sd, at, lossCurve, epochTimeMillis, new EvaluationRecord(trainingEvaluations));
    }

    /**
     * See {@link Listener#epochEnd(SameDiff, At, LossCurve, long)}, also provided the requested evaluations
     */
    public ListenerResponse epochEndEvaluations(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis, EvaluationRecord evaluations) {
        //No op
        return ListenerResponse.CONTINUE;
    }

    @Override
    public final ListenerResponse validationDone(SameDiff sd, At at, long validationTimeMillis) {
        return validationDoneEvaluations(sd, at, validationTimeMillis, new EvaluationRecord(validationEvaluations));
    }

    /**
     * See {@link Listener#validationDone(SameDiff, At, long)}, also provided the requested evaluations
     */
    public ListenerResponse validationDoneEvaluations(SameDiff sd, At at, long validationTimeMillis, EvaluationRecord evaluations) {
        //No op
        return ListenerResponse.CONTINUE;
    }

    @Override
    public final void activationAvailable(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, String varName,
            INDArray activation) {
        if(at.operation() == Operation.TRAINING) {
            if (trainingEvaluations.containsKey(varName)) {
                INDArray labels = batch.getLabels(evaluations().trainEvaluationLabels().get(varName));
                INDArray mask = batch.getLabelsMaskArray(evaluations().trainEvaluationLabels().get(varName));

                for (IEvaluation e : trainingEvaluations.get(varName))
                    e.eval(labels, activation, mask);
            }
        } else if(at.operation() == Operation.TRAINING_VALIDATION) {
            if (validationEvaluations.containsKey(varName)) {
                INDArray labels = batch.getLabels(evaluations().validationEvaluationLabels().get(varName));
                INDArray mask = batch.getLabelsMaskArray(evaluations().validationEvaluationLabels().get(varName));

                for (IEvaluation e : validationEvaluations.get(varName))
                    e.eval(labels, activation, mask);
            }
        }

        activationAvailableEvaluations(sd, at, batch, op, varName, activation);
    }

    /**
     * See {@link Listener#activationAvailable(SameDiff, At, MultiDataSet, SameDiffOp, String, INDArray)}
     */
    public void activationAvailableEvaluations(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, String varName,
            INDArray activation){
        //No op
    }

}
