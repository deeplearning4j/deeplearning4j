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

package org.nd4j.autodiff.listeners.impl;

import java.util.ArrayList;
import java.util.List;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseEvaluationListener;
import org.nd4j.autodiff.listeners.records.EvaluationRecord;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.listeners.ListenerEvaluations;
import org.nd4j.autodiff.listeners.ListenerResponse;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;

/**
 * HistoryListener is mainly used internally to collect information such as the loss curve and evaluations,
 * which will be reported later in a {@link History} instance
 */
public class HistoryListener extends BaseEvaluationListener {

    @Getter
    @Setter
    private ListenerEvaluations evaluations;

    private List<EvaluationRecord> trainingHistory = new ArrayList<>();
    private List<EvaluationRecord> validationHistory = new ArrayList<>();
    private LossCurve loss = null;

    private long startTime;
    private long endTime;

    private List<Long> validationTimes = new ArrayList<>();
    private long validationStartTime;


    public HistoryListener(TrainingConfig tc) {
        this.evaluations = new ListenerEvaluations(tc.getTrainEvaluations(), tc.getTrainEvaluationLabels(),
                tc.getValidationEvaluations(), tc.getValidationEvaluationLabels());
    }

    public HistoryListener(ListenerEvaluations evaluations) {
        this.evaluations = evaluations;
    }

    public HistoryListener newInstance() {
        return new HistoryListener(evaluations);
    }

    @Override
    public ListenerEvaluations evaluations() {
        return evaluations;
    }

    @Override
    public boolean isActive(Operation operation) {
        return operation.isTrainingPhase();
    }

    @Override
    public ListenerResponse epochEndEvaluations(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis, EvaluationRecord evaluations) {
        trainingHistory.add(evaluations);
        loss = lossCurve;

        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse validationDoneEvaluations(SameDiff sd, At at, long validationTimeMillis, EvaluationRecord evaluations) {
        validationHistory.add(evaluations);
        return ListenerResponse.CONTINUE;
    }

    @Override
    public void operationStart(SameDiff sd, Operation op) {
        if (op == Operation.TRAINING) {
            startTime = System.currentTimeMillis();
        } else if (op == Operation.TRAINING_VALIDATION) {
            validationStartTime = System.currentTimeMillis();
        }
    }

    @Override
    public void operationEnd(SameDiff sd, Operation op) {
        if (op == Operation.TRAINING) {
            endTime = System.currentTimeMillis();
        } else if (op == Operation.TRAINING_VALIDATION) {
            validationTimes.add(System.currentTimeMillis() - validationStartTime);
        }
    }

    public History getReport() {
        return new History(trainingHistory, validationHistory, loss, endTime - startTime, validationTimes);
    }

}
