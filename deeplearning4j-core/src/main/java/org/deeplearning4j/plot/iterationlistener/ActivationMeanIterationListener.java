package org.deeplearning4j.plot.iterationlistener;

import com.google.common.primitives.Ints;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.PlotFilters;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class ActivationMeanIterationListener implements IterationListener {
    private List<String> variables;
    private int iteration = 1;
    private PlotFilters filters = new PlotFilters();
    private File outputFile = new File("activations.png");

    public ActivationMeanIterationListener(List<String> variables) {
        this.variables = variables;
    }

    public ActivationMeanIterationListener(List<String> variables, int iteration) {
        this.variables = variables;
        this.iteration = iteration;
    }

    public List<String> getVariables() {
        return variables;
    }

    public void setVariables(List<String> variables) {
        this.variables = variables;
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }

    public PlotFilters getFilters() {
        return filters;
    }

    public void setFilters(PlotFilters filters) {
        this.filters = filters;
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
            INDArray weights = l.activationMean();
            if(weights.rank() < 4) {
                if(weights.rank() == 2) {
                    weights = weights.reshape(1,1, weights.rows(), weights.columns());
                }
                else if(weights.rank() == 3) {
                    weights = weights.reshape(Ints.concat(new int[]{1}, weights.shape()));
                }
            }

            INDArray plot = filters.render(weights, 1);
            BufferedImage image = ImageLoader.toBufferedImageRGB(plot);
            try {
                ImageIO.write(image, "png", outputFile);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
