package org.deeplearning4j.example.lfw;

import java.io.File;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.LFWDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawMnistGreyScale;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.rbm.CRBM;
import org.deeplearning4j.rbm.GaussianRectifiedLinearRBM;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LFWRBMExample {

    private static Logger log = LoggerFactory.getLogger(LFWRBMExample.class);
    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new LFWDataSetIterator(10,150000,40,40);
        log.info("Loading LFW...");
        DataSet all = iter.next(300);

        iter = new SamplingDataSetIterator(all,10,100);
        int cols = iter.inputColumns();
        log.info("Learning from " + cols);



        GaussianRectifiedLinearRBM r = new GaussianRectifiedLinearRBM.Builder()
                .numberOfVisible(iter.inputColumns())
                .useAdaGrad(true).withOptmizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .numHidden(900).withMomentum(3e-1)
                .normalizeByInputRows(true)
                .withLossFunction(LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .build();

        NeuralNetPlotter plotter = new NeuralNetPlotter();
        int numIter = 0;
        while(iter.hasNext()) {
            DataSet curr = iter.next();
            log.info("Training on pics " + curr.labelDistribution());
            r.trainTillConvergence(curr.getFirst(),1e-6,  new Object[]{1,1e-6,10000});
           /* if(numIter % 10 == 0) {
                FilterRenderer render = new FilterRenderer();
                try {
                    render.renderFilters(r.getW(), "currimg.png", (int)Math.sqrt(r.getW().rows) , (int) Math.sqrt( r.getW().rows));
                } catch (Exception e) {
                    log.error("Unable to plot filter, continuing...",e);
                }
            }
*/
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

                DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
                d.title = "REAL";
                d.draw();
                DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,1000,1000);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }


    }

}
