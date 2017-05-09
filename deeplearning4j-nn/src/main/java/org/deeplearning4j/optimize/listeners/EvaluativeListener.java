package org.deeplearning4j.optimize.listeners;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.*;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.concurrent.atomic.AtomicLong;

/**
 *
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class EvaluativeListener implements IterationListener {
    private ThreadLocal<AtomicLong> iterationCount = new ThreadLocal<>();
    private int frequency;

    private DataSetIterator dsIterator;
    private MultiDataSetIterator mdsIterator;
    private DataSet ds;
    private MultiDataSet mds;

    private IEvaluation[] evaluations;

    public EvaluativeListener(@NonNull DataSetIterator iterator, int frequency) {
        this(iterator, frequency, new Evaluation());
    }

    public EvaluativeListener(@NonNull MultiDataSetIterator iterator, int frequency) {
        this(iterator, frequency, new Evaluation());
    }

    public EvaluativeListener(@NonNull DataSetIterator iterator, int frequency, IEvaluation... evaluations) {
        this.dsIterator = iterator;
        this.frequency = frequency;
        this.evaluations = evaluations;
    }

    public EvaluativeListener(@NonNull MultiDataSetIterator iterator, int frequency, IEvaluation... evaluations) {
        this.mdsIterator = iterator;
        this.frequency = frequency;
        this.evaluations = evaluations;
    }

    public EvaluativeListener(@NonNull DataSet dataSet, int frequency) {
        this(dataSet, frequency, new Evaluation());
    }

    public EvaluativeListener(@NonNull MultiDataSet multiDataSet, int frequency) {
        this(multiDataSet, frequency, new Evaluation());
    }

    public EvaluativeListener(@NonNull DataSet dataSet, int frequency, IEvaluation... evaluations) {
        this.ds = dataSet;
        this.frequency = frequency;
        this.evaluations = evaluations;
    }

    public EvaluativeListener(@NonNull MultiDataSet multiDataSet, int frequency, IEvaluation... evaluations) {
        this.mds = multiDataSet;
        this.frequency = frequency;
        this.evaluations = evaluations;
    }

    /**
     * Get if listener invoked
     */
    @Override
    public boolean invoked() {
        return false;
    }

    /**
     * Change invoke to true
     */
    @Override
    public void invoke() {

    }

    /**
     * Event listener for each iteration
     *
     * @param model     the model iterating
     * @param iteration the iteration
     */
    @Override
    public void iterationDone(Model model, int iteration) {
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



        log.info("Starting evaluation on iteration {}", iteration);
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
                    evaluation.eval(ds.getLabels(), ((ComputationGraph) model).output(ds.getFeatureMatrix())[0]);
            } else if (mds != null) {
                for (IEvaluation evaluation : evaluations)
                    evaluation.eval(mds.getLabels()[0], ((ComputationGraph) model).output(mds.getFeatures())[0]);
            }
        } else throw new DL4JInvalidInputException("Model is unknown: " + model.getClass().getCanonicalName());

        // TODO: maybe something better should be used here?
        log.info("Reporting evaluation results:");
        for (IEvaluation evaluation: evaluations)
            log.info("{}:\n{}", evaluation.getClass().getSimpleName(), evaluation.stats());
    }
}
