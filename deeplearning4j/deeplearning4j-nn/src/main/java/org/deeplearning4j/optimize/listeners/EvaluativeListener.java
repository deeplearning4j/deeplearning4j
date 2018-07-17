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

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.callbacks.EvaluationCallback;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This TrainingListener implementation provides simple way for model evaluation during training.
 * It can be launched every Xth Iteration/Epoch, depending on frequency and InvocationType constructor arguments
 *
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class EvaluativeListener extends BaseTrainingListener {
    protected transient ThreadLocal<AtomicLong> iterationCount = new ThreadLocal<>();
    protected int frequency;

    protected AtomicLong invocationCount = new AtomicLong(0);

    protected transient DataSetIterator dsIterator;
    protected transient MultiDataSetIterator mdsIterator;
    protected DataSet ds;
    protected MultiDataSet mds;

    @Getter
    protected IEvaluation[] evaluations;

    @Getter
    protected InvocationType invocationType;

    /**
     * This callback will be invoked after evaluation finished
     */
    @Getter
    @Setter
    protected transient EvaluationCallback callback;

    /**
     * Evaluation will be launched after each *frequency* iteration
     * @param iterator
     * @param frequency
     */
    public EvaluativeListener(@NonNull DataSetIterator iterator, int frequency) {
        this(iterator, frequency, InvocationType.ITERATION_END, new Evaluation());
    }

    public EvaluativeListener(@NonNull DataSetIterator iterator, int frequency, @NonNull InvocationType type) {
        this(iterator, frequency, type, new Evaluation());
    }

    /**
     * Evaluation will be launched after each *frequency* iteration
     * @param iterator
     * @param frequency
     */
    public EvaluativeListener(@NonNull MultiDataSetIterator iterator, int frequency) {
        this(iterator, frequency, InvocationType.ITERATION_END, new Evaluation());
    }

    public EvaluativeListener(@NonNull MultiDataSetIterator iterator, int frequency, @NonNull InvocationType type) {
        this(iterator, frequency, type, new Evaluation());
    }

    /**
     * Evaluation will be launched after each *frequency* iteration
     *
     * @param iterator
     * @param frequency
     */
    public EvaluativeListener(@NonNull DataSetIterator iterator, int frequency, IEvaluation... evaluations) {
        this(iterator, frequency, InvocationType.ITERATION_END, evaluations);
    }

    public EvaluativeListener(@NonNull DataSetIterator iterator, int frequency, @NonNull InvocationType type,
                    IEvaluation... evaluations) {
        this.dsIterator = iterator;
        this.frequency = frequency;
        this.evaluations = evaluations;

        this.invocationType = type;
    }

    /**
     * Evaluation will be launched after each *frequency* iteration
     * @param iterator
     * @param frequency
     */
    public EvaluativeListener(@NonNull MultiDataSetIterator iterator, int frequency, IEvaluation... evaluations) {
        this(iterator, frequency, InvocationType.ITERATION_END, evaluations);
    }

    public EvaluativeListener(@NonNull MultiDataSetIterator iterator, int frequency, @NonNull InvocationType type,
                    IEvaluation... evaluations) {
        this.mdsIterator = iterator;
        this.frequency = frequency;
        this.evaluations = evaluations;

        this.invocationType = type;
    }

    public EvaluativeListener(@NonNull DataSet dataSet, int frequency, @NonNull InvocationType type) {
        this(dataSet, frequency, type, new Evaluation());
    }

    public EvaluativeListener(@NonNull MultiDataSet multiDataSet, int frequency, @NonNull InvocationType type) {
        this(multiDataSet, frequency, type, new Evaluation());
    }

    public EvaluativeListener(@NonNull DataSet dataSet, int frequency, @NonNull InvocationType type,
                    IEvaluation... evaluations) {
        this.ds = dataSet;
        this.frequency = frequency;
        this.evaluations = evaluations;

        this.invocationType = type;
    }

    public EvaluativeListener(@NonNull MultiDataSet multiDataSet, int frequency, @NonNull InvocationType type,
                    IEvaluation... evaluations) {
        this.mds = multiDataSet;
        this.frequency = frequency;
        this.evaluations = evaluations;

        this.invocationType = type;
    }

    /**
     * Event listener for each iteration
     *
     * @param model     the model iterating
     * @param iteration the iteration
     */
    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (invocationType == InvocationType.ITERATION_END)
            invokeListener(model);
    }

    @Override
    public void onEpochStart(Model model) {
        if (invocationType == InvocationType.EPOCH_START)
            invokeListener(model);
    }

    @Override
    public void onEpochEnd(Model model) {
        if (invocationType == InvocationType.EPOCH_END)
            invokeListener(model);
    }

    protected void invokeListener(Model model) {
        if (iterationCount.get() == null)
            iterationCount.set(new AtomicLong(0));

        if (iterationCount.get().getAndIncrement() % frequency != 0)
            return;

        for (IEvaluation evaluation : evaluations)
            evaluation.reset();

        if (dsIterator != null && dsIterator.resetSupported())
            dsIterator.reset();
        else if (mdsIterator != null && mdsIterator.resetSupported())
            mdsIterator.reset();

        // FIXME: we need to save/restore inputs, if we're being invoked with iterations > 1

        log.info("Starting evaluation nr. {}", invocationCount.incrementAndGet());
        if (model instanceof MultiLayerNetwork) {
            if (dsIterator != null) {
                ((MultiLayerNetwork) model).doEvaluation(dsIterator, evaluations);
            } else if (ds != null) {
                for (IEvaluation evaluation : evaluations)
                    evaluation.eval(ds.getLabels(), ((MultiLayerNetwork) model).output(ds.getFeatureMatrix()));
            }
        } else if (model instanceof ComputationGraph) {
            if (dsIterator != null) {
                ((ComputationGraph) model).doEvaluation(dsIterator, evaluations);
            } else if (mdsIterator != null) {
                ((ComputationGraph) model).doEvaluation(mdsIterator, evaluations);
            } else if (ds != null) {
                for (IEvaluation evaluation : evaluations)
                    evalAtIndex(evaluation, new INDArray[] {ds.getLabels()},
                                    ((ComputationGraph) model).output(ds.getFeatureMatrix()), 0);
            } else if (mds != null) {
                for (IEvaluation evaluation : evaluations)
                    evalAtIndex(evaluation, mds.getLabels(), ((ComputationGraph) model).output(mds.getFeatures()), 0);
            }
        } else
            throw new DL4JInvalidInputException("Model is unknown: " + model.getClass().getCanonicalName());

        // TODO: maybe something better should be used here?
        log.info("Reporting evaluation results:");
        for (IEvaluation evaluation : evaluations)
            log.info("{}:\n{}", evaluation.getClass().getSimpleName(), evaluation.stats());


        if (callback != null)
            callback.call(this, model, invocationCount.get(), evaluations);
    }

    protected void evalAtIndex(IEvaluation evaluation, INDArray[] labels, INDArray[] predictions, int index) {
        evaluation.eval(labels[index], predictions[index]);
    }

}
