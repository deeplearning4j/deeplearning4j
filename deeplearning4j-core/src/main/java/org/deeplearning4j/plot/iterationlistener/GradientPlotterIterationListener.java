package org.deeplearning4j.plot.iterationlistener;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;

/**
 * Plots weight distributions and activation probabilities
 *
 * @deprecated use {@link org.deeplearning4j.ui.weights.HistogramIterationListener} from deeplearning4j-ui instead
 */
@Deprecated
public class GradientPlotterIterationListener implements IterationListener {
    private int iterations = 10;
    private NeuralNetPlotter plotter = new NeuralNetPlotter();
    private boolean renderFirst = false;
    private boolean invoked = false;

    @Override
    public boolean invoked(){ return invoked; }

    @Override
    public void invoke() { this.invoked = true; }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param renderFirst render the graph on first pass
     */
    public GradientPlotterIterationListener(int iterations, boolean renderFirst) {
        this.iterations = iterations;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param plotter render the graph on first pass
     */
    public GradientPlotterIterationListener(int iterations, NeuralNetPlotter plotter) {
        this.iterations = iterations;
        this.plotter = plotter;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param plotter the plotter to use
     * @param renderFirst render the graph on first pass
     */
    public GradientPlotterIterationListener(int iterations, NeuralNetPlotter plotter, boolean renderFirst) {
        this.iterations = iterations;
        this.plotter = plotter;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     */
    public GradientPlotterIterationListener(int iterations) {
        this.iterations = iterations;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(iteration == 0 && renderFirst || iteration > 0 && iteration % this.iterations == 0) {
            this.invoke();
            Layer layer = (Layer) model;
            plotter.updateGraphDirectory(layer);
            plotter.plotNetworkGradient(layer,layer.gradient());
        }
    }
}
