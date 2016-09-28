package org.deeplearning4j.optimize.listeners.stats;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Alex on 28/09/2016.
 */
public class StatsListener implements IterationListener {

    private final StatsListenerReceiver receiver;
    private int iterCount = 0;



    public StatsListener(StatsListenerReceiver receiver){
        this.receiver = receiver;
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

        StatsListenerConfiguration config = receiver.getCurrentConfiguration();
        StatsReport report = receiver.newStatsReport();

        report.reportTime(System.currentTimeMillis());  //TODO optionally use NTP time source, based on configuration

        if(config.collectScore()){
            report.reportScore(model.score());
        }

        if(config.collectPerformanceStats()){

        }

        if(config.collectMeanMagnitudesParameters()){
            Map<String,Double> meanMagParams = calculateMeanMagnitudes(model.paramTable());
            report.reportMeanMagnitudesParameters(meanMagParams);
        }

        if(config.collectMeanMagnitudesUpdates()){
            Map<String,Double> meanMagUpdates = calculateMeanMagnitudes(model.gradient().gradientForVariable());
            report.reportMeanMagnitudesParameters(meanMagUpdates);
        }

        if(config.collectMeanMagnitudesActivations()){
            //TODO
        }



        iterCount++;
    }

    private static Map<String,Double> calculateMeanMagnitudes(Map<String,INDArray> source){
        Map<String,Double> meanMagnitudes = new HashMap<>();
        for(Map.Entry<String,INDArray> entry : source.entrySet()) {
            String name = entry.getKey();
            double meanMag = entry.getValue().norm1Number().doubleValue() / entry.getValue().length();
            meanMagnitudes.put(name, meanMag);
        }
        return meanMagnitudes;
    }
}
