package org.deeplearning4j.ui.plot.iterationlistener;

import org.canova.image.loader.ImageLoader;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.plot.PlotFilters;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class PlotFiltersIterationListener implements IterationListener {
    private List<String> variables;
    private int iteration = 1;
    private PlotFilters filters;
    private File outputFile = new File("render.png");


    public PlotFiltersIterationListener(PlotFilters plotFilters, List<String> variables, int iteration) {
        this.filters = plotFilters;
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
        if(this.iteration > 0 && iteration % this.iteration == 0) {
            INDArray weights = model.getParam(variables.get(0));
            filters.setInput(weights.transpose());
            filters.plot();
            INDArray plot = filters.getPlot();
            BufferedImage image = ImageLoader.toImage(plot);
            try {
                outputFile.createNewFile();
                ImageIO.write(image, "png", outputFile);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
