package org.deeplearning4j.optimize.listeners;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * This TrainingListener implementation provides a way to "sleep" during specific Neural Network training phases.
 * Suitable for debugging/testing purposes.
 *
 * PLEASE NOTE: All timers treat time values as milliseconds.
 * PLEASE NOTE: Do not use it in production environment.
 *
 * @author raver119@gmail.com
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
@Builder
public class SleepyTrainingListener implements TrainingListener {
    @Builder.Default protected long timerEpochEnd = 0L;
    @Builder.Default protected long timerEpochStart = 0L;
    @Builder.Default protected long timerFF = 0L;
    @Builder.Default protected long timerBP = 0L;
    @Builder.Default protected long timerIteration = 0L;

    @Override
    public void onEpochStart(Model model) {
        try {
            Thread.sleep(timerEpochStart);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void onEpochEnd(Model model) {
        try {
            Thread.sleep(timerEpochEnd);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        try {
            Thread.sleep(timerFF);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        try {
            Thread.sleep(timerFF);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void onGradientCalculation(Model model) {

    }

    @Override
    public void onBackwardPass(Model model) {
        try {
            Thread.sleep(timerBP);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
        try {
             Thread.sleep(timerIteration);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
