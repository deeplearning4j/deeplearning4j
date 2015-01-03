package org.deeplearning4j.scaleout.perform;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;

/**
 * Neural network work performer
 * @author Adam Gibson
 */
public class NeuralNetWorkPerformer implements WorkerPerformer {
    protected Layer neuralNetwork;

    public NeuralNetWorkPerformer() {

    }

    @Override
    public void perform(Job job) {
        Serializable work = job.getWork();
        if(work instanceof DataSet) {
            DataSet data = (DataSet) work;
            neuralNetwork.fit(data.getFeatureMatrix());
        }
        else if(work instanceof INDArray) {
            neuralNetwork.fit((INDArray) work);
        }

        job.setResult(neuralNetwork.params());


    }

    @Override
    public void update(Object... o) {
        INDArray arr = (INDArray) o[0];
        neuralNetwork.setParams(arr);

    }

    @Override
    public void setup(Configuration conf) {
        NeuralNetConfiguration conf2 = NeuralNetConfiguration.fromJson(conf.get(NEURAL_NET_CONF));
        this.neuralNetwork = conf2.getLayerFactory().create(conf2);
    }
}
