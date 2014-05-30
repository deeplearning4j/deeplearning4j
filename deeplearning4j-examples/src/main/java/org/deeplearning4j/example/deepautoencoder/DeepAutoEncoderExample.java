package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Demonstrates a DeepAutoEncoder reconstructions with
 * the MNIST digits
 */
public class DeepAutoEncoderExample {

    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample.class);

    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(80,80);

        Map<Integer,Boolean> activationForLayer = new HashMap<>();

        activationForLayer.put(3,true);


        DBN dbn = new DBN.Builder()
                .withHiddenUnitsByLayer(Collections.singletonMap(3, RBM.HiddenUnit.GAUSSIAN))
                .withLossFunction(NeuralNetwork.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .hiddenLayerSizes(new int[]{1000, 500, 250, 28}).withMomentum(0.9)
               // .lossFunctionByLayer(Collections.singletonMap(3, NeuralNetwork.LossFunction.SQUARED_LOSS))
                .withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .numberOfInputs(784)
                .numberOfOutPuts(2)
                .build();

        while(iter.hasNext()) {
            DataSet data = iter.next();
            data.filterAndStrip(new int[]{0,1});
            dbn.pretrain(data.getFirst(), new Object[]{1, 1e-1, 100});



        }







        iter.reset();





        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        encoder.setVisibleUnit(RBM.VisibleUnit.GAUSSIAN);
        encoder.setHiddenUnit(RBM.HiddenUnit.BINARY);
        while (iter.hasNext()) {
            DataSet next = iter.next();
            next.filterAndStrip(new int[]{0,1});
            log.info("Fine tune " + next.labelDistribution());
            encoder.finetune(next.getFirst(),1e-1,1000);

        }


        iter.reset();

        while (iter.hasNext()) {
            DataSet data = iter.next();
            data.filterAndStrip(new int[]{0,1});
            DoubleMatrix reconstruct = encoder.reconstruct(data.getFirst());
            for(int j = 0; j < data.numExamples(); j++) {

                DoubleMatrix draw1 = data.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                DoubleMatrix draw2 = reconstructed2.mul(255);

                DrawReconstruction d = new DrawReconstruction(draw1);
                d.title = "REAL";
                d.draw();
                DrawReconstruction d2 = new DrawReconstruction(draw2);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }


    }

}
