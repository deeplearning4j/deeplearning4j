package org.nd4j.autodiff.listeners.impl;

import lombok.NonNull;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.*;

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

    private MultiDataSet currentIterDataSet;

    private LogFileWriter writer;
    private boolean wroteLossNames;

    private Set<String> relevantOpsForEval;
    private Map<Pair<String,Integer>,Evaluation> epochTrainEval;
    boolean wroteEvalNames;



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

        Preconditions.checkState(!logFile.exists(), "Log file already exists: %s", logFile);
    }

    protected void initalizeWriter(SameDiff sd) {
        try{
            initializeHelper(sd);
        }catch (IOException e){
            throw new RuntimeException(e);
        }
    }

    protected void initializeHelper(SameDiff sd) throws IOException {
        writer = new LogFileWriter(logFile);

        //Write graph structure:
        writer.writeGraphStructure(sd);

        //Write system info:
        //TODO

        //All static info completed
        writer.writeFinishStaticMarker();
    }

    @Override
    public void epochStart(SameDiff sd, At at) {
        epochTrainEval = null;
    }

    @Override
    public void epochEnd(SameDiff sd, At at) {

        //If any evaluation, report it here:
        if(epochTrainEval != null){
            long time = System.currentTimeMillis();
            for(Map.Entry<Pair<String,Integer>,Evaluation> e : epochTrainEval.entrySet()){
                String n = "evaluation/" + e.getKey().getFirst();   //TODO what if user does same eval with multiple labels? Doesn't make sense... add validation to ensure this?

                List<Evaluation.Metric> l = trainEvalMetrics.get(e.getKey());
                for(Evaluation.Metric m : l) {
                    String mName = n + "/train/" + m.toString().toLowerCase();
                    if (!wroteEvalNames) {
                        writer.registerEventNameQuiet(mName);
                    }

                    double score = e.getValue().scoreForMetric(m);
                    try{
                        writer.writeScalarEvent(mName, LogFileWriter.EventSubtype.EVALUATION, time, at.iteration(), at.epoch(), score);
                    } catch (IOException ex){
                        throw new RuntimeException("Error writing to log file", ex);
                    }
                }

                wroteEvalNames = true;
            }
        }

        epochTrainEval = null;
    }

    @Override
    public void iterationStart(SameDiff sd, At at, MultiDataSet data, long etlMs) {
        //If there's any evaluation to do in opExecution method, we'll need this there
        currentIterDataSet = data;
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        if(writer == null)
            initalizeWriter(sd);
        long time = System.currentTimeMillis();

        //iterationDone method - just writes loss values (so far)

        if(!wroteLossNames){
            for(String s : loss.getLossNames()){
                writer.registerEventNameQuiet("losses/" + s);
            }

            if(loss.numLosses() > 1){
                writer.registerEventNameQuiet("losses/totalLoss");
            }
            wroteLossNames = true;
        }

        List<String> lossNames = loss.getLossNames();
        double[] lossVals = loss.getLosses();
        for( int i=0; i<lossVals.length; i++ ){
            try{
                String eventName = "losses/" + lossNames.get(i);
                writer.writeScalarEvent(eventName, LogFileWriter.EventSubtype.LOSS, time, at.iteration(), at.epoch(), lossVals[i]);
            } catch (IOException e){
                throw new RuntimeException("Error writing to log file", e);
            }
        }

        if(lossVals.length > 1){
            double total = loss.totalLoss();
            try{
                String eventName = "losses/totalLoss";
                writer.writeScalarEvent(eventName, LogFileWriter.EventSubtype.LOSS, time, at.iteration(), at.epoch(), total);
            } catch (IOException e){
                throw new RuntimeException("Error writing to log file", e);
            }
        }

        currentIterDataSet = null;
    }



    @Override
    public void opExecution(SameDiff sd, At at, SameDiffOp op, INDArray[] outputs) {


        //Do training set evaluation, if required
        //Note we'll do it in opExecution not iterationDone because we can't be sure arrays will be stil be around in the future
        //i.e., we'll eventually add workspaces and clear activation arrays once they have been consumed
        if(trainEvalMetrics != null && trainEvalMetrics.size() > 0){
            //First: check if this op is relevant at all to evaluation...
            if(relevantOpsForEval == null){
                //Build list for quick lookups to know if we should do anything for this op
                relevantOpsForEval = new HashSet<>();
                for (Pair<String, Integer> p : trainEvalMetrics.keySet()) {
                    Variable v = sd.getVariables().get(p.getFirst());
                    String opName = v.getOutputOfOp();
                    Preconditions.checkState(opName != null, "Cannot evaluate on variable of type %s - variable name: \"%s\"",
                            v.getVariable().getVariableType(), opName);
                    relevantOpsForEval.add(v.getOutputOfOp());
                }
            }

            if(!relevantOpsForEval.contains(op.getName())){
                //Op outputs are not required for eval
                return;
            }

            if(epochTrainEval == null) {
                epochTrainEval = new HashMap<>();

                for (Pair<String, Integer> p : trainEvalMetrics.keySet()) {
                    epochTrainEval.put(p, new Evaluation());
                }
            }

            //Perform evaluation:
            for (Pair<String, Integer> p : trainEvalMetrics.keySet()) {
                int idx = op.getOutputsOfOp().indexOf(p.getFirst());
                INDArray out = outputs[idx];
                INDArray label = currentIterDataSet.getLabels(p.getSecond());
                INDArray mask = currentIterDataSet.getLabelsMaskArray(p.getSecond());

                epochTrainEval.get(p).eval(label, out, mask);
            }
        }
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
        //TODO
    }
}
