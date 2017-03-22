package org.deeplearning4j.ui.flow.beans;

import lombok.Data;

import java.io.Serializable;
import java.util.*;

/**
 * @author raver119@gmail.com
 */
@Data
public class ModelState implements Serializable {
    private float score = 0.0f;
    private String trainingTime;
    private List<Float> scores = new ArrayList<>();
    private float performanceBatches;
    private float performanceSamples;
    private long iterationTime;

    private Map<String, Map> parameters = new HashMap<>();
    private Map<String, Map> gradients = new HashMap<>();
    private List<Double> learningRates = new ArrayList<>();

    private Map<Integer, LayerParams> layerParams = new LinkedHashMap<>();

    //private List<Float> performanceBatches = new ArrayList<>();
    //private List<Float> performanceSamples = new ArrayList<>();

    public ModelState() {

    }

    public void addScore(float score) {
        if (scores.size() > 1000)
            scores.remove(0);

        scores.add(score);
    }


    public void addPerformanceBatches(float perf) {
        performanceBatches = perf;
    }

    public void addPerformanceSamples(float perf) {
        performanceSamples = perf;
    }
    /*
    public void addPerformanceBatches(float perf) {
        if (performanceBatches.size() > 100)
            performanceBatches.remove(0);
    
        performanceBatches.add(perf);
    }
    
    public void addPerformanceSamples(float perf) {
        if (performanceSamples.size() > 100)
            performanceSamples.remove(0);
    
        performanceSamples.add(perf);
    }
    */
}
