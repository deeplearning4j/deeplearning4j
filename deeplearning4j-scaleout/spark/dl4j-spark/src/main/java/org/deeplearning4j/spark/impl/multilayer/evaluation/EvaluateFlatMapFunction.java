package org.deeplearning4j.spark.impl.multilayer.evaluation;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.IterativeReduceFlatMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**Function to evaluate data (classification), in a distributed manner
 * @author Alex Black
 */
public class EvaluateFlatMapFunction implements FlatMapFunction<Iterator<DataSet>, Evaluation> {

    private String json;
    private Broadcast<INDArray> params;
    private int examplesPerEvaluation;
    private static Logger log = LoggerFactory.getLogger(IterativeReduceFlatMap.class);

    /**
     * @param json Network configuration (json format)
     * @param params Network parameters
     * @param examplesPerEvaluation Max examples per evaluation. Do multiple separate forward passes if data exceeds
     *                              this. Used to avoid doing too many at once (and hence memory issues)
     */
    public EvaluateFlatMapFunction(String json, Broadcast<INDArray> params, int examplesPerEvaluation){
        this.json = json;
        this.params = params;
        this.examplesPerEvaluation = examplesPerEvaluation;
    }

    @Override
    public Iterable<Evaluation> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if (!dataSetIterator.hasNext()) {
            return Collections.singletonList(new Evaluation());
        }

        MultiLayerNetwork network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json));
        network.init();
        INDArray val = params.value();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcasted set parameters");
        network.setParameters(val);

        Evaluation evaluation = new Evaluation();
        List<DataSet> collect = new ArrayList<>();
        while (dataSetIterator.hasNext()) {
            collect.clear();
            while (dataSetIterator.hasNext() && collect.size() < examplesPerEvaluation) {
                collect.add(dataSetIterator.next());
            }

            DataSet data = DataSet.merge(collect, false);
            if (log.isDebugEnabled()) {
                log.debug("Evaluating {} examples ", data.numExamples());
            }

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
        }
        return Collections.singletonList(evaluation);
    }
}
