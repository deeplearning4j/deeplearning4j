package org.deeplearning4j.optimize.listeners;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Created by merlin on 6/29/15.
 */
public class ScorePlotterIterationListener implements IterationListener {
    private int iterations = 1;
    private NeuralNetPlotter plotter = new NeuralNetPlotter();
    private boolean renderFirst = false;
    private ArrayList<Double> scores = new ArrayList<>();
    private ArrayList<Double> accuracy = new ArrayList<>();
    private ArrayList<Double> weightUpdates = new ArrayList<>();

    /**
     *
     * @param iterations the number of iterations to render every plot
     */
    public ScorePlotterIterationListener(int iterations) {
        this.iterations = iterations;
    }

    protected String storeData(ArrayList data)  {
        try {
            String filePath = System.getProperty("java.io.tmpdir") + File.separator +  UUID.randomUUID().toString();
            File write = new File(filePath);
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            write.deleteOnExit();

            StringBuilder sb = new StringBuilder();
            for(Object value: data) {
                sb.append(String.format("%.10f", (Double) value));
                sb.append(",");
            }
            String line = sb.toString();
            line = line.substring(0, line.length()-1);
            bos.write(line.getBytes());
            bos.flush();
            bos.close();
            return filePath+",loss";

        } catch(IOException e){
            throw new RuntimeException(e);
        }
    }


    @Override
    public void iterationDone(Model model, int iteration) {
        scores.add(-model.score());
//        accuracy.add();
//        weightUpdates.add();

        // store
        if(iteration == 0 && renderFirst || iteration > 0 && iteration % this.iterations == 0) {
            String filePath = storeData(scores);
            plotter.plotLossFunction(filePath);
        }
    }

}
