package org.deeplearning4j.example.lfw;

import java.io.File;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LFWRBMExample {

    private static Logger log = LoggerFactory.getLogger(LFWRBMExample.class);
    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new LFWDataSetIterator(10,150000,28,28);
        log.info("Loading LFW...");
        DataSet all = iter.next(300);
        iter = new SamplingDataSetIterator(all,10,100);
        int cols = iter.inputColumns();
        log.info("Learning from " + cols);



        RBM r = new RBM.Builder()
                .withVisible(RBM.VisibleUnit.GAUSSIAN)
                .withHidden(RBM.HiddenUnit.RECTIFIED)
                .numberOfVisible(iter.inputColumns())
                .withOptmizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .numHidden(600).renderWeights(1)
                .withLossFunction(LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .build();

        NeuralNetPlotter plotter = new NeuralNetPlotter();
        int numIter = 0;
        /*
        Note that this is a demo meant to run faster, the faces learned are too fast, and the learning rate should be 1e-6
        or lower for proper learning to take place. Thi sis purely for demo purposes
         */
        while(iter.hasNext()) {
            DataSet curr = iter.next();
            curr.normalizeZeroMeanZeroUnitVariance();
            log.info("Training on pics " + curr.labelDistribution());
            r.trainTillConvergence(curr.getFirst(),1e-1,  new Object[]{1,1e-1,10000});
            if(numIter % 10 == 0) {
                FilterRenderer render = new FilterRenderer();
                try {
                    render.renderFilters(r.getW(), "currimg.png", (int)Math.sqrt(r.getW().rows) , (int) Math.sqrt( r.getW().rows),curr.numExamples());
                } catch (Exception e) {
                    log.error("Unable to plot filter, continuing...",e);
                }
            }

            numIter++;

        }
        File f = new File("faces-rbm.bin");
        log.info("Saving to " + f.getAbsolutePath());
        SerializationUtils.saveObject(r, f);
        iter.reset();






        //Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
        while(iter.hasNext()) {
            DataSet first = iter.next();
            DoubleMatrix reconstruct = r.reconstruct(first.getFirst());
            for(int j = 0; j < first.numExamples(); j++) {

                DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
                DoubleMatrix reconstructed2 = reconstruct.getRow(j).div(255);
                MatrixUtil.scaleByMax(reconstructed2);
                DoubleMatrix draw2 = reconstructed2.mul(255);

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
