package org.nd4j.autodiff.listeners.impl;

import lombok.NonNull;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class UIListener extends BaseListener {

    public enum UpdateRatio {L2, MEAN_MAGNITUDE}
    public enum HistogramType {PARAMETERS, PARAMETER_GRADIENTS, PARAMETER_UPDATES, ACTIVATIONS, ACTIVATION_GRADIENTS}


    private File logFile;
    private int lossPlotFreq;
    private int performanceStatsFrequency;
    private int updateRatioFrequency;
    private UpdateRatio updateRatioType;
    private int histogramFrequency;
    private HistogramType[] histogramTypes;
    private int opProfileFrequency;
    private Map<Pair<String,Integer>, List<Evaluation.Metric>> trainEvalMetrics;
    private TestEvaluation testEvaluation;


    private UIListener(Builder b){
        logFile = b.logFile;
        lossPlotFreq = b.lossPlotFreq;
        performanceStatsFrequency = b.performanceStatsFrequency;
        updateRatioFrequency = b.updateRatioFrequency;
        updateRatioType = b.updateRatioType;
        histogramFrequency = b.histogramFrequency;
        histogramTypes = b.histogramTypes;
        opProfileFrequency = b.opProfileFrequency;
        trainEvalMetrics = b.trainEvalMetrics;
        testEvaluation = b.testEvaluation;
    }



































    public static Builder builder(File logFile){
        return new Builder(logFile);
    }

    public static class Builder {

        private File logFile;

        private int lossPlotFreq = 1;
        private int performanceStatsFrequency = -1;     //Disabled by default

        private int updateRatioFrequency = -1;          //Disabled by default
        private UpdateRatio updateRatioType = UpdateRatio.MEAN_MAGNITUDE;

        private int histogramFrequency = -1;            //Disabled by default
        private HistogramType[] histogramTypes;

        private int opProfileFrequency = -1;            //Disabled by default

        private Map<Pair<String,Integer>, List<Evaluation.Metric>> trainEvalMetrics;

        private TestEvaluation testEvaluation = null;

        public Builder(@NonNull File logFile){
            this.logFile = logFile;
        }

        public Builder plotLosses(int frequency){
            this.lossPlotFreq = frequency;
            return this;
        }

        public Builder performanceStats(int frequency){
            this.performanceStatsFrequency = frequency;
            return this;
        }

        public Builder trainEvaluationMetrics(String name, int labelIdx, Evaluation.Metric... metrics){
            if(trainEvalMetrics == null){
                trainEvalMetrics = new LinkedHashMap<>();
            }
            Pair<String,Integer> p = new Pair<>(name, labelIdx);
            if(!trainEvalMetrics.containsKey(p)){
                trainEvalMetrics.put(p, new ArrayList<Evaluation.Metric>());
            }
            List<Evaluation.Metric> l = trainEvalMetrics.get(p);
            for(Evaluation.Metric m : metrics){
                if(!l.contains(m)){
                    l.add(m);
                }
            }
            return this;
        }

        public Builder trainAccuracy(String name, int labelIdx){
            return trainEvaluationMetrics(name, labelIdx, Evaluation.Metric.ACCURACY);
        }

        public Builder trainF1(String name, int labelIdx){
            return trainEvaluationMetrics(name, labelIdx, Evaluation.Metric.F1);
        }

        public Builder updateRatios(int frequency){
            return updateRatios(frequency, UpdateRatio.MEAN_MAGNITUDE);
        }

        public Builder updateRatios(int frequency, UpdateRatio ratioType){
            this.updateRatioFrequency = frequency;
            this.updateRatioType = ratioType;
            return this;
        }

        public Builder histograms(int frequency, HistogramType... types){
            this.histogramFrequency = frequency;
            this.histogramTypes = types;
            return this;
        }

        public Builder profileOps(int frequency){
            this.opProfileFrequency = frequency;
            return this;
        }

        public Builder testEvaluation(TestEvaluation testEvalConfig){
            this.testEvaluation = testEvalConfig;
            return this;
        }

        public UIListener build(){
            return new UIListener(this);
        }
    }

    public static class TestEvaluation {

    }
}
