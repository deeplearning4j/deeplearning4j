/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.plot.iterationlistener;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;

/**
 * Renders network activations every n iterations
 * @author Adam Gibson
 *
 * @deprecated use {@link org.deeplearning4j.ui.weights.HistogramIterationListener} from deeplearning4j-ui instead
 */
@Deprecated
public class NeuralNetPlotterIterationListener implements IterationListener {
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
    public NeuralNetPlotterIterationListener(int iterations, boolean renderFirst) {
        this.iterations = iterations;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param patchesPerRow the number of patches per row for rendering filters
     */
    public NeuralNetPlotterIterationListener(int iterations, int patchesPerRow) {
        this(iterations,patchesPerRow,false);
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param patchesPerRow the number of patches per row for rendering filters
     * @param renderFirst render on first iteration
     */
    public NeuralNetPlotterIterationListener(int iterations, int patchesPerRow, boolean renderFirst) {
        this.iterations = iterations;
        this.patchesPerRow = patchesPerRow;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param plotter the plotter to use
     * @param patchesPerRow the number of patches per row for rendering filters
     */
    public NeuralNetPlotterIterationListener(int iterations, NeuralNetPlotter plotter, int patchesPerRow) {
        this(iterations,plotter,patchesPerRow,false);
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param plotter the plotter to use
     * @param patchesPerRow the number of patches per row for rendering filters
     */
    public NeuralNetPlotterIterationListener(int iterations, NeuralNetPlotter plotter, int patchesPerRow,boolean renderFirst) {
        this.iterations = iterations;
        this.plotter = plotter;
        this.patchesPerRow = patchesPerRow;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     */
    public NeuralNetPlotterIterationListener(int iterations) {
        this.iterations = iterations;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(iteration == 0 && renderFirst || iteration > 0 && iteration % this.iterations == 0) {
            this.invoke();
            Layer layer = (Layer) model;
            plotter.updateGraphDirectory(layer);
            plotter.plotNetworkGradient(layer,layer.gradient());
            plotter.renderFilter(layer, patchesPerRow);
        }
    }
}
