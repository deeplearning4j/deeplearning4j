package org.deeplearning4j.example.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.models.featuredetectors.da.DenoisingAutoEncoder;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.sampling.Sampling;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.plot.FilterRenderer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DenoisingAutoEncoderMnistExample {


    private static Logger log = LoggerFactory.getLogger(DenoisingAutoEncoderMnistExample.class);

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        DenoisingAutoEncoder autoEncoder = new DenoisingAutoEncoder.Builder()
                .withOptmizationAlgo(NeuralNetwork.OptimizationAlgorithm.GRADIENT_DESCENT)
                .numberOfVisible(784).numHidden(600).build();


        //batches of 10, 60000 examples total
        DataSetIterator iter = new MultipleEpochsIterator(50,new MnistDataSetIterator(10,30));
        while(iter.hasNext()) {
            DataSet next = iter.next();
            autoEncoder.train(next.getFeatureMatrix(),1e-1f,0.3f,10);


        }


        iter.reset();


        FilterRenderer render = new FilterRenderer();
        render.renderFilters(autoEncoder.getW(), "example-render.jpg", 28, 28,10);




        //Iterate over the data applyTransformToDestination after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            INDArray reconstruct = autoEncoder.transform(first.getFeatureMatrix());
            for(int j = 0; j < first.numExamples(); j++) {

                INDArray draw1 = first.get(j).getFeatureMatrix().mul(255);
                INDArray reconstructed2 = reconstruct.getRow(j);
                INDArray draw2 = Sampling.binomial(reconstructed2, 1, new MersenneTwister(123)).mul(255);

                DrawReconstruction d = new DrawReconstruction(draw1);
                d.title = "REAL";
                d.draw();
                DrawReconstruction d2 = new DrawReconstruction(draw2,1000,1000);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }

    }

}
