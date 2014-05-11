package org.deeplearning4j.example.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawMnistGreyScale;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RBMMnistExample {

    private static Logger log = LoggerFactory.getLogger(RBMMnistExample.class);

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        RBM r = new RBM.Builder()
                .numberOfVisible(784).renderWeights(100)
                .numHidden(600).useRegularization(false).withL2(2e-2)
                .build();

        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(10,50);

        while(iter.hasNext()) {
            DataSet next = iter.next();
            log.info(String.valueOf(next.labelDistribution()));
            r.trainTillConvergence(next.getFirst(), 1e-2, new Object[]{1, 1e-2, 5000});

        }






        iter.reset();





        //Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct = r.reconstruct(first.getFirst());
            for(int j = 0; j < first.numExamples(); j++) {

                DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

                DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
                d.title = "REAL";
                d.draw();
                DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }

    }

}
