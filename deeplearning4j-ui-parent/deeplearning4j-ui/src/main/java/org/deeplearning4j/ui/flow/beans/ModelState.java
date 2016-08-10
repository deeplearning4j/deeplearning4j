package org.deeplearning4j.ui.flow.beans;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
@Data
public class ModelState {
    private List<Float> scores = new ArrayList<>();
    private float performanceBatches;
    private float performanceSamples;
    private Map<String,Map> parameters;
    private Map<String,Map> gradients;

    public ModelState() {

    }

    public void addScore(float score) {
        if (scores.size() > 1000)
            scores.remove(0);

        scores.add(score);
    }
}
