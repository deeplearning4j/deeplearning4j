package org.arbiter.deeplearning4j.evaluator;

import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

public class DL4JClassificationEvaluator implements ModelEvaluator<MultiLayerNetwork,DataSetIterator,Evaluation> {
    @Override
    public Evaluation evaluateModel(MultiLayerNetwork model, DataProvider<DataSetIterator> dataProvider) {

        DataSetIterator iterator = dataProvider.testData(null);
        Evaluation eval = new Evaluation();
        while(iterator.hasNext()){
            DataSet ds = iterator.next();
            INDArray features = ds.getFeatures();
            INDArray labels = ds.getLabels();
            INDArray out = model.output(features);
            //TODO: This won't work for time series (RNNs) + for masking
            eval.eval(labels,out);
        }

        return eval;
    }
}
