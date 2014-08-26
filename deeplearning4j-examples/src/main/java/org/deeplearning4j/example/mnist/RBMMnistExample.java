package org.deeplearning4j.example.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.sampling.Sampling;
import org.deeplearning4j.rbm.RBM;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RBMMnistExample {

    private static Logger log = LoggerFactory.getLogger(RBMMnistExample.class);

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        RBM r = new RBM.Builder()
                .numberOfVisible(784)
                .numHidden(600).renderWeights(10)
                .build();

        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(10,10);

        while(iter.hasNext()) {
            DataSet next = iter.next();
            //log.info(String.valueOf(next.labelDistribution()));
            r.trainTillConvergence(next.getFeatureMatrix(), 1e-1f, new Object[]{1, 1e-1f, 5000});

        }

        iter.reset();

        //Iterate over the dataset after you're done training and show the two side by side (you have to drag the test image to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            INDArray reconstruct = r.reconstruct(first.getFeatureMatrix());
            for(int j = 0; j < first.numExamples(); j++) {

                INDArray draw1 = first.get(j).getFeatureMatrix().mul(255);
                INDArray reconstructed2 = reconstruct.getRow(j);
                INDArray draw2 = Sampling.binomial(reconstructed2, 1, new MersenneTwister(123)).mul(255);

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
