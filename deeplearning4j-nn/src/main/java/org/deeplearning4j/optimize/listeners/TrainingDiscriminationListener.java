package org.deeplearning4j.optimize.listeners;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A listener that collects statistics important for evaluating the discrimination of
 * a training process. This includes the curveArea under a ROC curve developed from key points
 * during a training process. Statistics collected here are meant to be compared offline
 * for assessment.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TrainingDiscriminationListener implements TrainingListener {
    private final int frequency;
    private final boolean reportRocArea;
    private final boolean reportScore;

    private int xCount;
    private List<Double> curveX;
    private List<Double> curveY;
    private AtomicDouble curveArea = new AtomicDouble(0.0);

    public TrainingDiscriminationListener() {
        this(1, false, true);
    }

    public TrainingDiscriminationListener(int frequency) {
        this(frequency, false, true);
    }

    public TrainingDiscriminationListener(int frequency, boolean reportScore, boolean reportRocArea) {
        this.frequency = frequency;
        this.reportRocArea = reportRocArea;
        this.reportScore = reportScore;
        this.curveX = new ArrayList<>();
        this.curveY = new ArrayList<>();
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
        if(iteration % frequency == 0) {
            ++this.xCount;

            // each score is treated as a point in a polygon
            // for the purpose of calculating curveArea
            curveX.add((double) xCount);
            curveY.add(model.score());

            if(reportRocArea) {
                // add bottom right coordinates
                // copy the array so we don't pollute primary array with bottom coordinates
                List<Double> pointsX = new ArrayList<>(curveX);
                pointsX.add((double) xCount);
                List<Double> pointsY = new ArrayList<>(curveY);
                pointsY.add(0.0);

                curveArea.set(calculateArea(pointsX, pointsY, pointsX.size()));
                log.info("Score curve area at iteration " + iteration + " is " + curveArea.get());
            }
            if(reportScore) log.info("Score at iteration " + iteration + " is " + model.score());
        }
    }

    private double calculateArea(List<Double> pointsX, List<Double> pointsY, int nPoints) {
        double sum = 0;
        for (int i = 0; i < nPoints-1; i++) {
            sum += pointsX.get(i)*pointsY.get(i+1) - pointsY.get(i)*pointsX.get(i+1);
        }

        return Math.abs(sum / 2);
    }

    public void onEpochStart(Model var1) {
        // no op
    }

    public void onEpochEnd(Model var1) {
        // no op
    }

    public void onForwardPass(Model var1, List<INDArray> var2) {
        // no op
    }

    public void onForwardPass(Model var1, Map<String, INDArray> var2) {
        // no op
    }

    public void onGradientCalculation(Model var1) {
        // no op
    }

    public void onBackwardPass(Model var1) {
        // no op
    }
}
