package org.deeplearning4j.example.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.da.DenoisingAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;

public class DenoisingAutoEncoderMnistExample {

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        DenoisingAutoEncoder autoEncoder = new DenoisingAutoEncoder.Builder()
                .withSparsity(1e-1).renderWeights(1)
                .numberOfVisible(784).numHidden(600).build();


        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(10,30);
        while(iter.hasNext()) {
            DataSet next = iter.next();
            //train with k = 1 0.01 learning rate and 1000 epochs
            autoEncoder.trainTillConvergence(next.getFirst(), 1e-1, new Object[]{0.6,1e-1,1000});


        }


        iter.reset();


        FilterRenderer render = new FilterRenderer();
        render.renderFilters(autoEncoder.getW(), "example-render.jpg", 28, 28);




        //Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct = autoEncoder.reconstruct(first.getFirst());
            for(int j = 0; j < first.numExamples(); j++) {

                DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j);
                DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

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
