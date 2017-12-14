package org.deeplearning4j.optimize.listeners.checkpoint;

import com.google.common.io.Files;
import lombok.NonNull;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class CheckpointListener extends BaseTrainingListener {

    private enum KeepMode {ALL, LAST, LAST_AND_EVERY};

    private File rootDir;
    private KeepMode keepMode;
    private int keepLast;
    private int keepEvery;

    private int lastCheckpointNum = -1;
    private File checkpointRecordFile;

    private CheckpointListener(Builder builder){
        this.rootDir = builder.rootDir;
        this.keepMode = builder.keepMode;
        this.keepLast = builder.keepLast;
        this.keepEvery = builder.keepEvery;

        //TODO see if existing checkpoints are present
        this.checkpointRecordFile = new File(rootDir, "checkpointInfo.txt");
    }

    @Override
    public void onEpochEnd(Model model) {

    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        //Check
    }

    private void saveCheckpoint(Model model) {
        try{
            saveCheckpointHelper(model);
        } catch (Exception e){
            throw new RuntimeException("Error saving checkpoint", e);
        }
    }

    private void saveCheckpointHelper(Model model) throws Exception {
        Checkpoint c = new Checkpoint(lastCheckpointNum++, System.currentTimeMillis(), getIter(model), getEpoch(model),
                getModelType(model), null);
        setFileName(c);

        ModelSerializer.writeModel(model, new File(rootDir, c.getFilename()), true);

        String s = c.toFileString();
        write(s + "\n", checkpointRecordFile);
    }

    private static void setFileName(Checkpoint c){
        String filename = "checkpoint_" + c.getCheckpointNum() + "_" + c.getModelType() + ".zip";
        c.setFilename(filename);
    }

    public String write(String str, File f){
        try {
            if(!f.exists()){
                f.createNewFile();
            }
            Files.append(str, f, Charset.defaultCharset());
        } catch (IOException e){
            throw new RuntimeException(e);
        }
        return str;
    }

    protected static int getIter(Model model) {
        if (model instanceof MultiLayerNetwork) {
            return ((MultiLayerNetwork) model).getLayerWiseConfigurations().getIterationCount();
        } else if (model instanceof ComputationGraph) {
            return ((ComputationGraph) model).getConfiguration().getIterationCount();
        } else {
            return model.conf().getIterationCount();
        }
    }

    protected static int getEpoch(Model model) {
        if (model instanceof MultiLayerNetwork) {
            return ((MultiLayerNetwork) model).getLayerWiseConfigurations().getEpochCount();
        } else if (model instanceof ComputationGraph) {
            return ((ComputationGraph) model).getConfiguration().getEpochCount();
        } else {
            return model.conf().getEpochCount();
        }
    }

    protected static String getModelType(Model model){
        return model.getClass().getSimpleName();
    }


    public List<Checkpoint> availableCheckpoints(){
        if(!checkpointRecordFile.exists()){
            return Collections.emptyList();
        }
        List<String> lines;
        try(InputStream is = new BufferedInputStream(new FileInputStream(checkpointRecordFile))){
            lines = IOUtils.readLines(is);
        } catch (IOException e){
            throw new RuntimeException("Error loading checkpoint data from file: " + checkpointRecordFile.getAbsolutePath(), e);
        }

        List<Checkpoint> out = new ArrayList<>(lines.size()-1); //Assume first line is header
        for( int i=1; i<lines.size(); i++ ){
            out.add(Checkpoint.fromFileString(lines.get(i)));
        }

        return out;
    }

    public File getFileForCheckpoint(Checkpoint checkpoint){
        return getFileForCheckpoint(checkpoint.getCheckpointNum());
    }

    public File getFileForCheckpoint(int checkpointNum){

        throw new UnsupportedOperationException("Not yet implemented");
    }

    public MultiLayerNetwork loadCheckpointMLN(Checkpoint checkpoint){
        return loadCheckpointMLN(checkpoint.getCheckpointNum());
    }

    public MultiLayerNetwork loadCheckpointMLN(int checkpointNum){

        throw new UnsupportedOperationException("Not yet implemented");
    }

    public ComputationGraph loadCheckpointCG(Checkpoint checkpoint){
        return loadCheckpointCG(checkpoint.getCheckpointNum());
    }

    public ComputationGraph loadCheckpointCG(int checkpointNum){

        throw new UnsupportedOperationException("Not yet implemented");
    }

    public static class Builder {

        private File rootDir;
        private KeepMode keepMode;
        private int keepLast;
        private int keepEvery;
        private boolean logSaving;


        public Builder(@NonNull String rootDir){
            this(new File(rootDir));
        }

        public Builder(@NonNull File rootDir){
            this.rootDir = rootDir;
        }

        public Builder keepAll(){
            this.keepMode = KeepMode.ALL;
            return this;
        }

        public Builder keepLast(int n){
            this.keepMode = KeepMode.LAST;
            this.keepLast = n;
            return this;
        }

        public Builder keepLastAndEvery(int nLast, int everyN){
            this.keepMode = KeepMode.LAST_AND_EVERY;
            this.keepLast = nLast;
            this.keepEvery = everyN;
            return this;
        }

        public Builder logSaving(boolean logSaving){
            this.logSaving = logSaving;
            return this;
        }


    }

}
