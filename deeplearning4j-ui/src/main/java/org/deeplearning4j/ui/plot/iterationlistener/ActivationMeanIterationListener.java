package org.deeplearning4j.ui.plot.iterationlistener;

import org.canova.image.loader.ImageLoader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * @author Adam Gibson
 */
public class ActivationMeanIterationListener implements IterationListener {
    private int iteration = 1;
    private File outputFile = new File("activations.png");


    public ActivationMeanIterationListener(int iteration) {
        this.iteration = iteration;
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }


    public File getOutputFile() {
        return outputFile;
    }

    public void setOutputFile(File outputFile) {
        this.outputFile = outputFile;
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {

    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(iteration % this.iteration == 0) {
            Layer l = (Layer) model;
            INDArray activationMean = l.activate();
            INDArray weights = Transforms.sigmoid(activationMean);


            BufferedImage image = ImageLoader.toImage(weights);
            try {
                ImageIO.write(image, "png", outputFile);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
