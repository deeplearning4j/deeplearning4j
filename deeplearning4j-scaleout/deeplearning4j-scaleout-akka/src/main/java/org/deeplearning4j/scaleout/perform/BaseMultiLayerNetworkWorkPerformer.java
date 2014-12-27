package org.deeplearning4j.scaleout.perform;

import org.deeplearning4j.nn.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;

/**
 * Work performer for a base multi layer network
 * @author Adam Gibson
 */
public class BaseMultiLayerNetworkWorkPerformer implements WorkerPerformer {
    private MultiLayerNetwork multiLayerNetwork;


    @Override
    public void setup(Configuration conf) {
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(conf.get(MULTI_LAYER_CONF));
        multiLayerNetwork = new MultiLayerNetwork(conf2);
    }

    @Override
    public void perform(Job job) {
        Serializable work = job.getWork();
        if(work instanceof DataSet) {
            DataSet data = (DataSet) work;
            multiLayerNetwork.fit(data);
            job.setResult(multiLayerNetwork.params());
        }
    }

    @Override
    public void update(Object... o) {
        INDArray arr = (INDArray) o[0];
        multiLayerNetwork.setParams(arr);

    }
}
