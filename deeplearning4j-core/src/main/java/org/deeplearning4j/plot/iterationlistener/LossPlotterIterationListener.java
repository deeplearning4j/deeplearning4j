package org.deeplearning4j.plot.iterationlistener;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Reference: https://cs231n.github.io/neural-networks-3/
 */

public class LossPlotterIterationListener implements IterationListener {
    private int iterations = 1;
    private NeuralNetPlotter plotter = new NeuralNetPlotter();
    private boolean renderFirst = false;
    private ArrayList<Double> scores = new ArrayList<>();
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
    public LossPlotterIterationListener(int iterations, boolean renderFirst) {
        this.iterations = iterations;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     * @param plotter the plotter to use
     */
    public LossPlotterIterationListener(int iterations, NeuralNetPlotter plotter) {
        this.iterations = iterations;
        this.plotter = plotter;
    }


    /**
     *
     * @param iterations the number of iterations to render every
     * @param plotter the plotter to use
     * @param renderFirst render the graph on first pass
     */
    public LossPlotterIterationListener(int iterations, NeuralNetPlotter plotter, boolean renderFirst) {
        this.iterations = iterations;
        this.plotter = plotter;
        this.renderFirst = renderFirst;
    }

    /**
     *
     * @param iterations the number of iterations to render every
     */
    public LossPlotterIterationListener(int iterations) {
        this.iterations = iterations;
    }



    @Override
    public void iterationDone(Model model, int iteration) {
        scores.add(model.score());

        if (iteration == 0 && renderFirst || iteration > 0 && iteration % this.iterations == 0) {
            this.invoke();
            plotter.updateGraphDirectory((Layer) model);
            String dataFilePath = plotter.writeArray(scores);
            plotter.renderGraph("loss", dataFilePath, plotter.getLayerGraphFilePath() + "loss.png");
        }
    }

}
