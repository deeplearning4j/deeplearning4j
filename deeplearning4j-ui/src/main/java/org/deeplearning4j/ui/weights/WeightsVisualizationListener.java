package org.deeplearning4j.ui.weights;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
public class WeightsVisualizationListener implements IterationListener {
    private int freq = 10;
    private static final Logger log = LoggerFactory.getLogger(WeightsVisualizationListener.class);
    private int minibatchNum = 0;

    public WeightsVisualizationListener() {

    }

    public WeightsVisualizationListener(int visualizationFrequency) {
        this.freq = visualizationFrequency;
    }

    /**
     * Get if listener invoked
     */
    @Override
    public boolean invoked() {
        return false;
    }

    /**
     * Change invoke to true
     */
    @Override
    public void invoke() {

    }

    /**
     * Event listener for each iteration
     *
     * @param model     the model iterating
     * @param iteration the iteration number
     */
    @Override
    public void iterationDone(Model model, int iteration) {
        if (iteration == 5) {

            log.info("Model param keys: " + model.paramTable().keySet());

            for (String key: model.paramTable().keySet()) {
                log.info("Dimensions for params at key ["+key+"] : " + Arrays.toString(model.paramTable().get(key).shape()));
                if (model.paramTable().get(key).shape().length == 2){
                    writeRows(model.paramTable().get(key), new File("mb_"+minibatchNum+"_param_" + key + ".txt"));
                }
            }

            Layer l = (Layer) model;
            INDArray activationMean = l.activate();

            log.info("Activation mean shape: " + Arrays.toString(activationMean.shape()));

            INDArray weights = Transforms.sigmoid(activationMean);

            log.info("Weights shape: " + Arrays.toString(weights.shape()));

           // log.info("weights: " + weights);
           //  log.info("activation mean: " + activationMean);
            writeRows(activationMean, new File("activationMean_"+minibatchNum+".txt"));
            writeRows(weights, new File("weights_"+minibatchNum+".txt"));

            BufferedImage image = ImageLoader.toImage(weights);
            BufferedImage image2 = ImageLoader.toImage(activationMean);
            try {
                ImageIO.write(image, "png", new File("tmp_"+minibatchNum+".png"));
                ImageIO.write(image2, "png", new File("mean_"+minibatchNum+".png"));
            } catch (IOException e) {
                e.printStackTrace();
            }

            //throw new RuntimeException("Early stop");
            minibatchNum++;
        }
    }


    private void writeRows(INDArray array, File file) {
        try {
            PrintWriter writer = new PrintWriter(file);
            for (int x = 0; x < array.rows(); x++) {
                writer.println("Row [" + x + "]: " + array.getRow(x));
            }
            writer.flush();
            writer.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
