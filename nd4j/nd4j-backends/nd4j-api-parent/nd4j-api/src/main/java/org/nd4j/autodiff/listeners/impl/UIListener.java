package org.nd4j.autodiff.listeners.impl;

import com.google.flatbuffers.Table;
import lombok.NonNull;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.ListenerResponse;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.graph.UIGraphStructure;
import org.nd4j.graph.UIInfoType;
import org.nd4j.graph.UIStaticInfoRecord;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * User interface listener for SameDiff<br>
 * <br>
 * <b>Basic usage:</b>
 * <pre>
 * {@code
 * UIListener l = UIListener.builder(f)
 *                  //Plot loss curve, at every iteration (enabled and set to 1 by default)
 *                 .plotLosses(1)
 *                 //Plot the training set evaluation metrics: accuracy and f1 score
 *                 .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
 *                 //Plot the parameter to update:ratios for each parameter, every 10 iterations
 *                 .updateRatios(10)
 *                 .build();
 * }
 * </pre>
 * <br>
 * Note that the UIListener supports continuing with the same network on the same file - but only if the network configuration
 * matches. See {@link FileMode} for configuration/details
 *
 * @author Alex Black
 */
public class UIListener extends BaseListener {

    /**
     * Default: FileMode.CREATE_OR_APPEND<br>
     * The mode for handling behaviour when an existing UI file already exists<br>
     * CREATE: Only allow new file creation. An exception will be thrown if the log file already exists.<br>
     * APPEND: Only allow appending to an existing file. An exception will be thrown if: (a) no file exists, or (b) the
     * network configuration in the existing log file does not match the current log file.<br>
     * CREATE_OR_APPEND: As per APPEND, but create a new file if none already exists.<br>
     * CREATE_APPEND_NOCHECK: As per CREATE_OR_APPEND, but no exception will be thrown if the existing model does not
     * match the current model structure. This mode is not recommended.<br>
     */
    public enum FileMode {CREATE, APPEND, CREATE_OR_APPEND, CREATE_APPEND_NOCHECK}

    /**
     * Used to specify how the Update:Parameter ratios are computed. Only relevant when the update ratio calculation is
     * enabled via {@link Builder#updateRatios(int, UpdateRatio)}; update ratio collection is disabled by default<br>
     * L2: l2Norm(updates)/l2Norm(parameters) is used<br>
     * MEAN_MAGNITUDE: mean(abs(updates))/mean(abs(parameters)) is used<br>
     */
    public enum UpdateRatio {L2, MEAN_MAGNITUDE}

    /**
     * Used to specify which histograms should be collected. Histogram collection is disabled by default, but can be
     * enabled via {@link Builder#histograms(int, HistogramType...)}. Note that multiple histogram types may be collected simultaneously.<br>
     * Histograms may be collected for:<br>
     * PARAMETERS: All trainable parameters<br>
     * PARAMETER_GRADIENTS: Gradients corresponding to the trainable parameters<br>
     * PARAMETER_UPDATES: All trainable parameter updates, before they are applied during training (updates are gradients after applying updater and learning rate etc)<br>
     * ACTIVATIONS: Activations - ARRAY type SDVariables - those that are not constants, variables or placeholders<br>
     * ACTIVATION_GRADIENTS: Activation gradients
     */
    public enum HistogramType {PARAMETERS, PARAMETER_GRADIENTS, PARAMETER_UPDATES, ACTIVATIONS, ACTIVATION_GRADIENTS}


    private FileMode fileMode;
    private File logFile;
    private int lossPlotFreq;
    private int performanceStatsFrequency;
    private int updateRatioFrequency;
    private UpdateRatio updateRatioType;
    private int histogramFrequency;
    private HistogramType[] histogramTypes;
    private int opProfileFrequency;
    private Map<Pair<String,Integer>, List<Evaluation.Metric>> trainEvalMetrics;
    private int trainEvalFrequency;
    private TestEvaluation testEvaluation;
    private int learningRateFrequency;

    private MultiDataSet currentIterDataSet;

    private LogFileWriter writer;
    private boolean wroteLossNames;
    private boolean wroteLearningRateName;

    private Set<String> relevantOpsForEval;
    private Map<Pair<String,Integer>,Evaluation> epochTrainEval;
    private boolean wroteEvalNames;
    private boolean wroteEvalNamesIter;

    private int firstUpdateRatioIter = -1;

    private boolean checkStructureForRestore;

    private UIListener(Builder b){
        fileMode = b.fileMode;
        logFile = b.logFile;
        lossPlotFreq = b.lossPlotFreq;
        performanceStatsFrequency = b.performanceStatsFrequency;
        updateRatioFrequency = b.updateRatioFrequency;
        updateRatioType = b.updateRatioType;
        histogramFrequency = b.histogramFrequency;
        histogramTypes = b.histogramTypes;
        opProfileFrequency = b.opProfileFrequency;
        trainEvalMetrics = b.trainEvalMetrics;
        trainEvalFrequency = b.trainEvalFrequency;
        testEvaluation = b.testEvaluation;
        learningRateFrequency = b.learningRateFrequency;

        switch (fileMode){
            case CREATE:
                Preconditions.checkState(!logFile.exists(), "Log file already exists and fileMode is set to CREATE: %s\n" +
                        "Either delete the existing file, specify a path that doesn't exist, or set the UIListener to another mode " +
                        "such as CREATE_OR_APPEND", logFile);
                break;
            case APPEND:
                Preconditions.checkState(logFile.exists(), "Log file does not exist and fileMode is set to APPEND: %s\n" +
                        "Either specify a path to an existing log file for this model, or set the UIListener to another mode " +
                        "such as CREATE_OR_APPEND", logFile);
                break;
        }

        if(logFile.exists())
            restoreLogFile();

    }

    protected void restoreLogFile(){
        if(logFile.length() == 0 && fileMode == FileMode.CREATE_OR_APPEND || fileMode == FileMode.APPEND){
            logFile.delete();
            return;
        }

        try {
            writer = new LogFileWriter(logFile);
        } catch (IOException e){
            throw new RuntimeException("Error restoring existing log file at path: " + logFile.getAbsolutePath(), e);
        }

        if(fileMode == FileMode.APPEND || fileMode == FileMode.CREATE_OR_APPEND){
            //Check the graph structure, if it exists.
            //This is to avoid users creating UI log file with one network configuration, then unintentionally appending data
            // for a completely different network configuration

            LogFileWriter.StaticInfo si;
            try {
                si = writer.readStatic();
            } catch (IOException e){
                throw new RuntimeException("Error restoring existing log file, static info at path: " + logFile.getAbsolutePath(), e);
            }

            List<Pair<UIStaticInfoRecord, Table>> staticList = si.getData();
            if(si != null) {
                for (int i = 0; i < staticList.size(); i++) {
                    UIStaticInfoRecord r = staticList.get(i).getFirst();
                    if (r.infoType() == UIInfoType.GRAPH_STRUCTURE){
                        //We can't check structure now (we haven't got SameDiff instance yet) but we can flag it to check on first iteration
                        checkStructureForRestore = true;
                    }
                }
            }

        }
    }

    protected void checkStructureForRestore(SameDiff sd){
        LogFileWriter.StaticInfo si;
        try {
            si = writer.readStatic();
        } catch (IOException e){
            throw new RuntimeException("Error restoring existing log file, static info at path: " + logFile.getAbsolutePath(), e);
        }

        List<Pair<UIStaticInfoRecord, Table>> staticList = si.getData();
        if(si != null) {
            UIGraphStructure structure = null;
            for (int i = 0; i < staticList.size(); i++) {
                UIStaticInfoRecord r = staticList.get(i).getFirst();
                if (r.infoType() == UIInfoType.GRAPH_STRUCTURE){
                    structure = (UIGraphStructure) staticList.get(i).getSecond();
                    break;
                }
            }

            if(structure != null){
                int nInFile = structure.inputsLength();
                List<String> phs = new ArrayList<>(nInFile);
                for( int i=0; i<nInFile; i++ ){
                    phs.add(structure.inputs(i));
                }

                List<String> actPhs = sd.inputs();
                if(actPhs.size() != phs.size() || !actPhs.containsAll(phs)){
                    throw new IllegalStateException("Error continuing collection of UI stats in existing model file " + logFile.getAbsolutePath() +
                            ": Model structure differs. Existing (file) model placeholders: " + phs + " vs. current model placeholders: " + actPhs +
                            ". To disable this check, use FileMode.CREATE_APPEND_NOCHECK though this may result issues when rendering data via UI");
                }

                //Check variables:
                int nVarsFile = structure.variablesLength();
                List<String> vars = new ArrayList<>(nVarsFile);
                for( int i=0; i<nVarsFile; i++ ){
                    vars.add(structure.variables(i).name());
                }
                List<SDVariable> sdVars = sd.variables();
                List<String> varNames = new ArrayList<>(sdVars.size());
                for(SDVariable v : sdVars){
                    varNames.add(v.name());
                }

                if(varNames.size() != vars.size() || !varNames.containsAll(vars)){
                    int countDifferent = 0;
                    List<String> different = new ArrayList<>();
                    for(String s : varNames){
                        if(!vars.contains(s)){
                            countDifferent++;
                            if(different.size() < 10){
                                different.add(s);
                            }
                        }
                    }
                    StringBuilder msg = new StringBuilder();
                    msg.append("Error continuing collection of UI stats in existing model file ")
                            .append(logFile.getAbsolutePath())
                            .append(": Current model structure differs vs. model structure in file - ").append(countDifferent).append(" variable names differ.");
                    if(different.size() == countDifferent){
                        msg.append("\nVariables in new model not present in existing (file) model: ").append(different);
                    } else {
                        msg.append("\nFirst 10 variables in new model not present in existing (file) model: ").append(different);
                    }
                    msg.append("\nTo disable this check, use FileMode.CREATE_APPEND_NOCHECK though this may result issues when rendering data via UI");

                    throw new IllegalStateException(msg.toString());
                }
            }
        }

        checkStructureForRestore = false;
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
    public boolean isActive(Operation operation) {
        return operation == Operation.TRAINING;
    }

    @Override
    public void epochStart(SameDiff sd, At at) {
        epochTrainEval = null;
    }

    @Override
    public ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis) {

        //If any training evaluation, report it here:
        if(epochTrainEval != null){
            long time = System.currentTimeMillis();
            for(Map.Entry<Pair<String,Integer>,Evaluation> e : epochTrainEval.entrySet()){
                String n = "evaluation/" + e.getKey().getFirst();   //TODO what if user does same eval with multiple labels? Doesn't make sense... add validation to ensure this?

                List<Evaluation.Metric> l = trainEvalMetrics.get(e.getKey());
                for(Evaluation.Metric m : l) {
                    String mName = n + "/train/" + m.toString().toLowerCase();
                    if (!wroteEvalNames) {
                        if(!writer.registeredEventName(mName)) {    //Might have been registered if continuing training
                            writer.registerEventNameQuiet(mName);
                        }
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
        return ListenerResponse.CONTINUE;
    }

    @Override
    public void iterationStart(SameDiff sd, At at, MultiDataSet data, long etlMs) {
        if(writer == null)
            initalizeWriter(sd);
        if(checkStructureForRestore)
            checkStructureForRestore(sd);

        //If there's any evaluation to do in opExecution method, we'll need this there
        currentIterDataSet = data;
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        long time = System.currentTimeMillis();

        //iterationDone method - just writes loss values (so far)

        if(!wroteLossNames){
            for(String s : loss.getLossNames()){
                String n = "losses/" + s;
                if(!writer.registeredEventName(n)) {    //Might have been registered if continuing training
                    writer.registerEventNameQuiet(n);
                }
            }

            if(loss.numLosses() > 1){
                String n = "losses/totalLoss";
                if(!writer.registeredEventName(n)) {    //Might have been registered if continuing training
                    writer.registerEventNameQuiet(n);
                }
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

        if(learningRateFrequency > 0){
            //Collect + report learning rate
            if(!wroteLearningRateName){
                String name = "learningRate";
                if(!writer.registeredEventName(name)) {
                    writer.registerEventNameQuiet(name);
                }
                wroteLearningRateName = true;
            }

            if(at.iteration() % learningRateFrequency == 0) {
                IUpdater u = sd.getTrainingConfig().getUpdater();
                if (u.hasLearningRate()) {
                    double lr = u.getLearningRate(at.iteration(), at.epoch());
                    try {
                        writer.writeScalarEvent("learningRate", LogFileWriter.EventSubtype.LEARNING_RATE, time, at.iteration(), at.epoch(), lr);
                    } catch (IOException e){
                        throw new RuntimeException("Error writing to log file");
                    }
                }
            }
        }
    }



    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {


        //Do training set evaluation, if required
        //Note we'll do it in opExecution not iterationDone because we can't be sure arrays will be stil be around in the future
        //i.e., we'll eventually add workspaces and clear activation arrays once they have been consumed
        if(at.operation() == Operation.TRAINING && trainEvalMetrics != null && trainEvalMetrics.size() > 0){
            long time = System.currentTimeMillis();

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
            boolean wrote = false;
            for (Pair<String, Integer> p : trainEvalMetrics.keySet()) {
                int idx = op.getOutputsOfOp().indexOf(p.getFirst());
                INDArray out = outputs[idx];
                INDArray label = currentIterDataSet.getLabels(p.getSecond());
                INDArray mask = currentIterDataSet.getLabelsMaskArray(p.getSecond());

                epochTrainEval.get(p).eval(label, out, mask);

                if(trainEvalFrequency > 0 && at.iteration() > 0 && at.iteration() % trainEvalFrequency == 0){
                    for(Evaluation.Metric m : trainEvalMetrics.get(p)) {
                        String n = "evaluation/train_iter/" + p.getKey() + "/" + m.toString().toLowerCase();
                        if (!wroteEvalNamesIter) {
                            if(!writer.registeredEventName(n)) {    //Might have been written previously if continuing training
                                writer.registerEventNameQuiet(n);
                            }
                            wrote = true;
                        }

                        double score = epochTrainEval.get(p).scoreForMetric(m);

                        try {
                            writer.writeScalarEvent(n, LogFileWriter.EventSubtype.EVALUATION, time, at.iteration(), at.epoch(), score);
                        } catch (IOException e) {
                            throw new RuntimeException("Error writing to log file");
                        }
                    }
                }
            }
            wroteEvalNamesIter = wrote;
        }
    }

    @Override
    public void preUpdate(SameDiff sd, At at, Variable v, INDArray update) {
        if(writer == null)
            initalizeWriter(sd);

        if(updateRatioFrequency > 0 && at.iteration() % updateRatioFrequency == 0){
            if(firstUpdateRatioIter < 0){
                firstUpdateRatioIter = at.iteration();
            }

            if(firstUpdateRatioIter == at.iteration()){
                //Register name
                String name = "logUpdateRatio/" + v.getName();
                if(!writer.registeredEventName(name)){  //Might have already been registered if continuing
                    writer.registerEventNameQuiet(name);
                }
            }

            double params;
            double updates;
            if(updateRatioType == UpdateRatio.L2){
                params = v.getVariable().getArr().norm2Number().doubleValue();
                updates = update.norm2Number().doubleValue();
            } else {
                //Mean magnitude - L1 norm divided by N. But in the ratio later, N cancels out...
                params = v.getVariable().getArr().norm1Number().doubleValue();
                updates = update.norm1Number().doubleValue();
            }

            double ratio = updates / params;
            if(params == 0.0){
                ratio = 0.0;
            } else {
                ratio = Math.max(-10, Math.log10(ratio));   //Clip to -10, when updates are too small
            }


            try{
                String name = "logUpdateRatio/" + v.getName();
                writer.writeScalarEvent(name, LogFileWriter.EventSubtype.LOSS, System.currentTimeMillis(), at.iteration(), at.epoch(), ratio);
            } catch (IOException e){
                throw new RuntimeException("Error writing to log file", e);
            }
        }
    }





    public static Builder builder(File logFile){
        return new Builder(logFile);
    }

    public static class Builder {

        private FileMode fileMode = FileMode.CREATE_OR_APPEND;
        private File logFile;

        private int lossPlotFreq = 1;
        private int performanceStatsFrequency = -1;     //Disabled by default

        private int updateRatioFrequency = -1;          //Disabled by default
        private UpdateRatio updateRatioType = UpdateRatio.MEAN_MAGNITUDE;

        private int histogramFrequency = -1;            //Disabled by default
        private HistogramType[] histogramTypes;

        private int opProfileFrequency = -1;            //Disabled by default

        private Map<Pair<String,Integer>, List<Evaluation.Metric>> trainEvalMetrics;
        private int trainEvalFrequency = 10;            //Report evaluation metrics every 10 iterations by default

        private TestEvaluation testEvaluation = null;

        private int learningRateFrequency = 10;         //Whether to plot learning rate or not

        public Builder(@NonNull File logFile){
            this.logFile = logFile;
        }

        public Builder fileMode(FileMode fileMode){
            this.fileMode = fileMode;
            return this;
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

        public Builder trainEvalFrequency(int trainEvalFrequency){
            this.trainEvalFrequency = trainEvalFrequency;
            return this;
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

        public Builder learningRate(int frequency){
            this.learningRateFrequency = frequency;
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
