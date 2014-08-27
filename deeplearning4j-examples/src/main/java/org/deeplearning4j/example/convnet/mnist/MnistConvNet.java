package org.deeplearning4j.example.convnet.mnist;

import cc.mallet.util.FileUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.models.featuredetectors.rbm.ConvolutionalRBM;
import org.deeplearning4j.util.ImageLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 *
 */
public class MnistConvNet {

    private static Logger log = LoggerFactory.getLogger(MnistConvNet.class);

    public static void main(String[] args) throws Exception {
        RandomGenerator gen = new MersenneTwister(123);

        int rows = 28;
        int cols = 28;

        double fanIn = 28 * 28;
        double abs = Math.sqrt(6 / fanIn);
        ConvolutionalRBM r = new ConvolutionalRBM
                .Builder().withFilterSize(new int[]{7, 7})
                .withNumFilters(new int[]{9,9}).withStride(new int[]{2, 2}).withVisibleSize(new int[]{rows,cols}).withLossFunction(NeuralNetwork.LossFunction.NEGATIVELOGLIKELIHOOD)
                .withOptmizationAlgo(NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT).withRandom(gen)
                .withSparsity(5e-2f).withSparseGain(5)
                .numberOfVisible(28).numHidden(28)
                .withMomentum(0.5f)
                .build();


        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(1,200);
        //DataSetIterator iter = new LFWDataSetIterator(1,150000,rows,cols);

        while(iter.hasNext()) {
            DataSet next = iter.next();
            log.info("Len " + next.getFeatureMatrix().length());
            INDArray reshape = next.getFeatureMatrix().reshape(rows,cols);
            INDArray W = (INDArray) r.getW();
            log.info("W shape " + W.shape());
            r.fit(reshape, 5e-2f, new Object[]{1, 5e-2f, 20});


        };


        FileUtils.writeObject(new File("mnist-conv.ser"),r);
        drawFilters(r);


    }



    public static void drawSample(ConvolutionalRBM r,int rows, int cols,INDArray input) throws Exception {

        INDArray draw = input.dup();
        DrawReconstruction greyScale = new DrawReconstruction(input);
        greyScale.readjustToData();
        greyScale.draw();

        log.info("Draw sum " + draw.sum(Integer.MAX_VALUE));
        BufferedImage img = ImageLoader.toImage(input);
        File write = new File("newtmpfile-pool.png");

        Graphics2D g = img.createGraphics();
        g.drawImage(img, 0, 0, cols,rows, null);
        g.setComposite(AlphaComposite.Src);

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);



        g.dispose();



        ImageIO.write(img, "png", write);

    }

    public static void drawFilters(ConvolutionalRBM r) throws Exception {
        INDArray W =  r.getW().dup();

        INDArray draw =  W.reshape(W.rows() * W.columns(),W.slices());
        draw.muli(255);
        FilterRenderer render = new FilterRenderer();
        BufferedImage img = render.renderFilters(draw,"tmpfile.png",draw.rows(),draw.columns(),W.slices());
        BufferedImage resizedImage = new BufferedImage(49, 49, img.getType());
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(img, 0, 0, 49, 49, null);
        g.setComposite(AlphaComposite.Src);

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);



        g.dispose();
        File write = new File("newtmpfile.png");
        if(write.exists())
            write.delete();
        ImageIO.write(resizedImage, "png", write);

    }

}
