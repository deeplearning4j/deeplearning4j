package org.deeplearning4j.optimizer.listener;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.checkpoint.CheckpointListener;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

import static org.junit.Assert.assertEquals;

public class TestCheckpointListener {

    @Rule
    public TemporaryFolder tempDir = new TemporaryFolder();

    @Test
    public void testCheckpointListener1() throws Exception {
        File f = tempDir.newFolder();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        CheckpointListener l = new CheckpointListener.Builder(f)
                .keepAll()
                .saveEveryNEpochs(2)
                .build();

        net.setListeners(l);

        DataSetIterator iter = new IrisDataSetIterator(75,150);
        for(int i=0; i<10; i++ ){
            net.fit(iter);

            if(i > 0 && i % 2 == 0){
                assertEquals(1 + i/2, f.list().length);
            }
        }

        //Expect models saved at end of epochs: 1, 3, 5, 7... (i.e., after 2, 4, 6 etc epochs)
        File[] files = f.listFiles();
        int count = 0;
        for(File f2 : files){
            if(!f2.getPath().endsWith(".zip")){
                continue;
            }
            count++;

            int prefixLength = "checkpoint_".length();
            int num = Integer.parseInt(f2.getName().substring(prefixLength, prefixLength+1));

            MultiLayerNetwork n = ModelSerializer.restoreMultiLayerNetwork(f2, true);
            int expEpoch = 2 * (num + 1) - 1;   //Saved at the end of the previous epoch
            int expIter = (expEpoch+1) * 2;     //+1 due to epochs being zero indexed

            assertEquals(expEpoch, n.getEpochCount());
            assertEquals(expIter, n.getIterationCount());
        }

    }

}
