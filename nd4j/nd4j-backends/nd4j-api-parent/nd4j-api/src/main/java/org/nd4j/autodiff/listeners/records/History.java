/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.autodiff.listeners.records;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import lombok.Getter;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.IMetric;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * An object containing training history for a SameDiff.fit call, such as {@link SameDiff#fit()}, {@link SameDiff#fit(DataSetIterator, int, Listener...)}, etc.<br>
 * Contains information including:<br>
 * - Evaluations performed (training set and test set)<br>
 * - Loss curve (score values at each iteration)<br>
 * - Training times, and validation times<br>
 * - Number of epochs performed<br>
 */
@Getter
public class History {

    private List<EvaluationRecord> trainingHistory;
    private List<EvaluationRecord> validationHistory;

    private LossCurve lossCurve;

    private long trainingTimeMillis;
    private List<Long> validationTimesMillis;

    public History(List<EvaluationRecord> training, List<EvaluationRecord> validation, LossCurve loss,
            long trainingTimeMillis, List<Long> validationTimesMillis){
        trainingHistory = Collections.unmodifiableList(training);
        validationHistory = Collections.unmodifiableList(validation);
        this.lossCurve = loss;
        this.trainingTimeMillis = trainingTimeMillis;
        this.validationTimesMillis = Collections.unmodifiableList(validationTimesMillis);
    }

    /**
     * Get the training evaluations
     */
    public List<EvaluationRecord> trainingEval(){
        return trainingHistory;
    }

    /**
     * Get the validation evaluations
     */
    public List<EvaluationRecord> validationEval(){
        return validationHistory;
    }

    /**
     * Get the loss curve
     */
    public LossCurve lossCurve(){
        return lossCurve;
    }

    /**
     * Get the total training time, in milliseconds
     */
    public long trainingTimeMillis(){
        return trainingTimeMillis;
    }

    /**
     * Get the total validation time, in milliseconds
     */
    public List<Long> validationTimesMillis(){
        return validationTimesMillis;
    }

    /**
     * Get the number of epochs trained for
     */
    public int trainingEpochs(){
        return trainingHistory.size();
    }

    /**
     * Get the number of epochs validation was ran on
     */
    public int validationEpochs(){
        return validationHistory.size();
    }

    /**
     * Get the results of a training evaluation on a given parameter for a given metric
     *
     * Only works if there is only one evaluation with the given metric for param
     */
    public List<Double> trainingEval(String param, IMetric metric){
        List<Double> data = new ArrayList<>();
        for(EvaluationRecord er : trainingHistory)
            data.add(er.getValue(param, metric));

        return data;
    }

    /**
     * Get the results of a training evaluation on a given parameter for a given metric
     *
     * Only works if there is only one evaluation with the given metric for param
     */
    public List<Double> trainingEval(SDVariable param, IMetric metric){
        return trainingEval(param.name(), metric);
    }

    /**
     * Get the results of a training evaluation on a given parameter at a given index, for a given metric
     *
     * Note that it returns all recorded evaluations.
     * Index determines the evaluation used not the epoch's results to return.
     */
    public List<Double> trainingEval(String param, int index, IMetric metric){
        List<Double> data = new ArrayList<>();
        for(EvaluationRecord er : trainingHistory)
            data.add(er.getValue(param, index, metric));

        return data;
    }

    /**
     * Get the results of a training evaluation on a given parameter at a given index, for a given metric
     *
     * Note that it returns all recorded evaluations.
     * Index determines the evaluation used not the epoch's results to return.
     */
    public List<Double> trainingEval(SDVariable param, int index, IMetric metric){
        return trainingEval(param.name(), index, metric);
    }

    /**
     * Get the results of a training evaluation for a given metric
     *
     * Only works if there is only one evaluation with the given metric
     */
    public List<Double> trainingEval(IMetric metric){
        List<Double> data = new ArrayList<>();
        for(EvaluationRecord er : trainingHistory)
            data.add(er.getValue(metric));

        return data;
    }

    /**
     * Get the results of a training evaluation on a given parameter
     *
     * Only works if there is only one evaluation for param.
     */
    public List<IEvaluation> trainingEval(String param){
        List<IEvaluation> data = new ArrayList<>();
        for(EvaluationRecord er : trainingHistory)
            data.add(er.evaluation(param));

        return data;
    }

    /**
     * Get the results of a training evaluation on a given parameter
     *
     * Only works if there is only one evaluation for param.
     */
    public List<IEvaluation> trainingEval(SDVariable param){
        return trainingEval(param.name());
    }

    /**
     * Get the results of a training evaluation on a given parameter at a given index
     *
     * Note that it returns all recorded evaluations.
     * Index determines the evaluation used not the epoch's results to return.
     */
    public List<IEvaluation> trainingEval(String param, int index){
        List<IEvaluation> data = new ArrayList<>();
        for(EvaluationRecord er : trainingHistory)
            data.add(er.evaluation(param, index));

        return data;
    }

    /**
     * Get the results of a training evaluation on a given parameter at a given index
     *
     * Note that it returns all recorded evaluations.
     * Index determines the evaluation used not the epoch's results to return.
     */
    public List<IEvaluation> trainingEval(SDVariable param, int index){
        return trainingEval(param.name(), index);
    }

    /**
     * Get the results of a validation evaluation on a given parameter for a given metric
     *
     * Only works if there is only one evaluation with the given metric for param
     */
    public List<Double> validationEval(String param, IMetric metric){
        List<Double> data = new ArrayList<>();
        for(EvaluationRecord er : validationHistory)
            data.add(er.getValue(param, metric));

        return data;
    }

    /**
     * Get the results of a validation evaluation on a given parameter for a given metric
     *
     * Only works if there is only one evaluation with the given metric for param
     */
    public List<Double> validationEval(SDVariable param, IMetric metric){
        return validationEval(param.name(), metric);
    }

    /**
     * Get the results of a validation evaluation on a given parameter at a given index, for a given metric
     *
     * Note that it returns all recorded evaluations.
     * Index determines the evaluation used not the epoch's results to return.
     */
    public List<Double> validationEval(String param, int index, IMetric metric){
        List<Double> data = new ArrayList<>();
        for(EvaluationRecord er : validationHistory)
            data.add(er.getValue(param, index, metric));

        return data;
    }

    /**
     * Get the results of a validation evaluation on a given parameter at a given index, for a given metric
     *
     * Note that it returns all recorded evaluations.
     * Index determines the evaluation used not the epoch's results to return.
     */
    public List<Double> validationEval(SDVariable param, int index, IMetric metric){
        return validationEval(param.name(), index, metric);
    }

    /**
     * Get the results of a validation evaluation for a given metric
     *
     * Only works if there is only one evaluation with the given metric
     */
    public List<Double> validationEval(IMetric metric){
        List<Double> data = new ArrayList<>();
        for(EvaluationRecord er : validationHistory)
            data.add(er.getValue(metric));

        return data;
    }

    /**
     * Get the results of a validation evaluation on a given parameter
     *
     * Only works if there is only one evaluation for param.
     */
    public List<IEvaluation> validationEval(String param){
        List<IEvaluation> data = new ArrayList<>();
        for(EvaluationRecord er : validationHistory)
            data.add(er.evaluation(param));

        return data;
    }

    /**
     * Get the results of a validation evaluation on a given parameter
     *
     * Only works if there is only one evaluation for param.
     */
    public List<IEvaluation> validationEval(SDVariable param){
        return validationEval(param.name());
    }

    /**
     * Get the results of a validation evaluation on a given parameter at a given index
     *
     * Note that it returns all recorded evaluations.
     * Index determines the evaluation used not the epoch's results to return.
     */
    public List<IEvaluation> validationEval(String param, int index){
        List<IEvaluation> data = new ArrayList<>();
        for(EvaluationRecord er : validationHistory)
            data.add(er.evaluation(param, index));

        return data;
    }

    /**
     * Get the results of a validation evaluation on a given parameter at a given index
     *
     * Note that it returns all recorded evaluations.
     * Index determines the evaluation used not the epoch's results to return.
     */
    public List<IEvaluation> validationEval(SDVariable param, int index){
        return validationEval(param.name(), index);
    }

    /**
     * Gets the training evaluations ran during the last epoch
     */
    public EvaluationRecord finalTrainingEvaluations(){
        Preconditions.checkState(!trainingHistory.isEmpty(), "Cannot get final training evaluation - history is empty");
        return trainingHistory.get(trainingHistory.size() - 1);
    }

    /**
     * Gets the validation evaluations ran during the last epoch
     */
    public EvaluationRecord finalValidationEvaluations(){
        Preconditions.checkState(!validationHistory.isEmpty(), "Cannot get final validation evaluation - history is empty");
        return validationHistory.get(validationHistory.size() - 1);
    }

    /**
     * Gets the evaluation record for a given epoch.
     * @param epoch The epoch to get results for.  If negative, returns results for the epoch that many epochs from the end.
     */
    public EvaluationRecord trainingEvaluations(int epoch){
        if(epoch >= 0){
            return trainingHistory.get(epoch);
        } else {
            return trainingHistory.get(trainingHistory.size() - epoch);
        }
    }

    /**
     * Gets the evaluation record for a given epoch.
     * @param epoch The epoch to get results for.  If negative, returns results for the epoch that many epochs from the end.
     */
    public EvaluationRecord validationEvaluations(int epoch){
        if(epoch >= 0){
            return trainingHistory.get(epoch);
        } else {
            return validationHistory.get(validationHistory.size() - epoch);
        }
    }

}
