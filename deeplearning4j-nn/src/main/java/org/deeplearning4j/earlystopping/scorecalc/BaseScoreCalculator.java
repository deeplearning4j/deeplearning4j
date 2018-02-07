package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public abstract class BaseScoreCalculator<T extends Model> implements ScoreCalculator<T> {

    protected DataSetIterator iterator;
    protected double scoreSum;
    protected int minibatchCount;
    protected int exampleCount;

    protected BaseScoreCalculator(DataSetIterator iterator){
        this.iterator = iterator;
    }

    @Override
    public double calculateScore(T network) {
        reset();

        if(!iterator.hasNext())
            iterator.reset();

        while(iterator.hasNext()){
            DataSet ds = iterator.next();
            INDArray out = output(network, ds.getFeatures());
            scoreSum += scoreMinibatch(network, ds.getFeatures(), ds.getLabels(), out);
            minibatchCount++;
            exampleCount += ds.getFeatures().size(0);
        }

        return finalScore(scoreSum, minibatchCount, exampleCount);
    }

    protected abstract void reset();

    protected abstract INDArray output(T network, INDArray input);

    protected abstract double scoreMinibatch(T network, INDArray features, INDArray labels, INDArray output);

    protected abstract double finalScore(double scoreSum, int minibatchCount, int exampleCount);
}
