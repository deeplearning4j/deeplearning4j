package org.deeplearning4j.plot.iterationlistener;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;

/**
 * @deprecated try to use deeplearning4j-ui instead
 */
@Deprecated
public class RenderFilterIterationListener implements IterationListener {
    private int iterations = 10;
    private NeuralNetPlotter plotter = new NeuralNetPlotter();
    private int patchesPerRow = 100;
    private boolean renderFirst = false;
    private boolean invoked = false;

    @Override
    public boolean invoked(){ return invoked; }

    @Override
    public void invoke() { this.invoked = true; }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param renderFirst render on first iteration
     **/
    public RenderFilterIterationListener(int iterations, boolean renderFirst) {
        this.iterations = iterations;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param patchesPerRow the number of patches per row for rendering filters
     */
    public RenderFilterIterationListener(int iterations, int patchesPerRow) {
        this(iterations,patchesPerRow,false);
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param patchesPerRow the number of patches per row for rendering filters
     * @param renderFirst render the graph on first pass
     */
    public RenderFilterIterationListener(int iterations, int patchesPerRow, boolean renderFirst) {
        this.iterations = iterations;
        this.patchesPerRow = patchesPerRow;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param plotter the plotter to use
     * @param patchesPerRow the number of patches per row for rendering filters
     * @param renderFirst render the graph on first pass
     */
    public RenderFilterIterationListener(int iterations, NeuralNetPlotter plotter, int patchesPerRow, boolean renderFirst) {
        this.iterations = iterations;
        this.plotter = plotter;
        this.patchesPerRow = patchesPerRow;
        this.renderFirst = renderFirst;
    }


    /**
     *
     * @param iterations the number of iterations to render every
     */
    public RenderFilterIterationListener(int iterations) {
        this.iterations = iterations;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if (iteration == 0 && renderFirst || iteration > 0 && iteration % this.iterations == 0) {
            this.invoke();
            Layer layer = (Layer) model;
            plotter.renderFilter(layer, patchesPerRow);
        }


    }
}
