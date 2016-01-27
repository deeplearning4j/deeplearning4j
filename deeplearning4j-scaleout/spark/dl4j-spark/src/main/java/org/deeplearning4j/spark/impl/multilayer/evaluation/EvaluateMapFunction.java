package org.deeplearning4j.spark.impl.multilayer.evaluation;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Function to evaluate data (classification), in a distributed manner
 * @author Alex Black
 */

public class EvaluateMapFunction implements Function<DataSet, Evaluation> {

    protected static Logger log = LoggerFactory.getLogger(EvaluateMapFunction.class);

    protected MultiLayerNetwork network;
    protected List<String> labels = new ArrayList<>();

    public EvaluateMapFunction(MultiLayerNetwork network, List<String> labels) {
        this.network = network;
        this.labels = labels;
    }

    @Override
    public Evaluation call(DataSet data) throws Exception {
        Evaluation evaluation = new Evaluation(labels);

        INDArray out;
        if(data.hasMaskArrays()) {
            out = network.output(data.getFeatureMatrix(), false, data.getFeaturesMaskArray(), data.getLabelsMaskArray());
        } else {
            out = network.output(data.getFeatureMatrix(), false);
        }

        if(data.getLabels().rank() == 3){
            if(data.getLabelsMaskArray() == null){
                evaluation.evalTimeSeries(data.getLabels(),out);
            } else {
                evaluation.evalTimeSeries(data.getLabels(),out,data.getLabelsMaskArray());
            }
        } else {
            evaluation.eval(data.getLabels(),out);
        }
        return evaluation;

    }
}
