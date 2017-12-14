package org.deeplearning4j.optimize.listeners.checkpoint;

import com.google.common.io.Files;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
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
import java.util.*;
import java.util.concurrent.TimeUnit;

@Slf4j
public class CheckpointListener extends BaseTrainingListener {

    private enum KeepMode {ALL, LAST, LAST_AND_EVERY};

    private File rootDir;
    private KeepMode keepMode;
    private int keepLast;
    private int keepEvery;
    private boolean logSaving;

    private Integer saveEveryNEpochs;
    private Integer saveEveryNIterations;
    private boolean saveEveryNIterSinceLast;
    private Long saveEveryAmount;
    private TimeUnit saveEveryUnit;
    private Long saveEveryMs;
    private boolean saveEverySinceLast;

    private int lastCheckpointNum = -1;
    private File checkpointRecordFile;

    private Checkpoint lastCheckpoint;
    private long startTime = -1;
    private int startIter = -1;
    private Long lastSaveEveryMsNoSinceLast;

    private CheckpointListener(Builder builder){
        this.rootDir = builder.rootDir;
        this.keepMode = builder.keepMode;
        this.keepLast = builder.keepLast;
        this.keepEvery = builder.keepEvery;
        this.logSaving = builder.logSaving;

        this.saveEveryNEpochs = builder.saveEveryNEpochs;
        this.saveEveryNIterations = builder.saveEveryNIterations;
        this.saveEveryNIterSinceLast = builder.saveEveryNIterSinceLast;
        this.saveEveryAmount = builder.saveEveryAmount;
        this.saveEveryUnit = builder.saveEveryUnit;
        this.saveEverySinceLast = builder.saveEverySinceLast;

        if(saveEveryAmount != null){
            saveEveryMs = TimeUnit.MILLISECONDS.convert(saveEveryAmount, saveEveryUnit);
        }

        //TODO see if existing checkpoints are present
        this.checkpointRecordFile = new File(rootDir, "checkpointInfo.txt");
    }

    @Override
    public void onEpochEnd(Model model) {
        int epochsDone = getEpoch(model) + 1;
        if(saveEveryNEpochs != null && epochsDone > 0 && epochsDone % saveEveryNEpochs == 0){
            //Save:
            saveCheckpoint(model);
        }
        //General saving conditions: don't need to check here - will check in iterationDone
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (startTime < 0) {
            startTime = System.currentTimeMillis();
            startIter = iteration;
            return;
        }

        //Check iterations saving condition:
        if(saveEveryNIterations != null){
            if(saveEveryNIterSinceLast){
                //Consider last saved model when deciding whether to save
                long lastSaveIter = (lastCheckpoint != null ? lastCheckpoint.getIteration() : startIter);
                if(iteration - lastSaveIter >= saveEveryNIterations){
                    saveCheckpoint(model);
                    return;
                }
            } else {
                //Same every N iterations, regardless of saving time
                if(iteration > 0 && iteration % saveEveryNIterations == 0){
                    saveCheckpoint(model);
                    return;
                }
            }
        }

        //Check time saving condition:
        long time = System.currentTimeMillis();
        if(saveEveryUnit != null){
            if(saveEverySinceLast){
                //Consider last saved when when deciding whether to save
                long lastSaveTime = (lastCheckpoint != null ? lastCheckpoint.getTimestamp() : startTime);
                if((time - lastSaveTime) >= saveEveryMs){
                    saveCheckpoint(model);
                    return;
                }
            } else {
                //Save periodically, regardless of when last model was saved
                long lastSave = (lastSaveEveryMsNoSinceLast != null ? lastSaveEveryMsNoSinceLast : startTime);
                if((time - lastSave) > saveEveryMs){
                    saveCheckpoint(model);
                    lastSaveEveryMsNoSinceLast = time;
                    return;
                }
            }
        }
    }

    private void saveCheckpoint(Model model) {
        try{
            saveCheckpointHelper(model);
        } catch (Exception e){
            throw new RuntimeException("Error saving checkpoint", e);
        }
    }

    private void saveCheckpointHelper(Model model) throws Exception {
        Checkpoint c = new Checkpoint(++lastCheckpointNum, System.currentTimeMillis(), getIter(model), getEpoch(model),
                getModelType(model), null);
        setFileName(c);

        ModelSerializer.writeModel(model, new File(rootDir, c.getFilename()), true);

        String s = c.toFileString();
        write(s + "\n", checkpointRecordFile);

        if(logSaving){
            log.info("Model checkpoint saved: epoch {}, iteration {}, path: {}", c.getEpoch(), c.getIteration(),
                    new File(rootDir, c.getFilename()).getPath() );
        }
        this.lastCheckpoint = c;


        //Finally: determine if we should delete some old models...
        if(keepMode == null || keepMode == KeepMode.ALL){
            return;
        } else if(keepMode == KeepMode.LAST){
            List<Checkpoint> checkpoints = availableCheckpoints();
            Iterator<Checkpoint> iter = checkpoints.iterator();
            while(checkpoints.size() > keepLast){
                Checkpoint toRemove = iter.next();
                File f = getFileForCheckpoint(toRemove);
                f.delete();
                iter.remove();
            }
        } else {
            //Keep mode: last N and every M
            for(Checkpoint cp : availableCheckpoints()){
                if(cp.getCheckpointNum() > 0 && cp.getCheckpointNum() % keepEvery == 0){
                    //One of the "every M to keep" models
                    continue;
                } else if(cp.getCheckpointNum() > lastCheckpointNum - keepLast ){        //Example: latest is 5, keep last 2 -> keep checkpoints 4 and 5
                    //One of last N to keep
                    continue;
                }
                //Otherwise: delete file
                File f = getFileForCheckpoint(cp);
                f.delete();
            }
        }
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
            Checkpoint c = Checkpoint.fromFileString(lines.get(i));
            if(new File(rootDir, c.getFilename()).exists()){
                out.add(c);
            }
        }
        return out;
    }

    public Checkpoint lastCheckpoint(){
        List<Checkpoint> all = availableCheckpoints();
        if(all.size() == 0){
            return null;
        }
        return all.get(all.size()-1);
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
        private boolean logSaving = true;

        private Integer saveEveryNEpochs;
        private Integer saveEveryNIterations;
        private boolean saveEveryNIterSinceLast;
        private Long saveEveryAmount;
        private TimeUnit saveEveryUnit;
        private boolean saveEverySinceLast;


        public Builder(@NonNull String rootDir){
            this(new File(rootDir));
        }

        public Builder(@NonNull File rootDir){
            this.rootDir = rootDir;
        }

        public Builder saveEveryEpoch(){
            return saveEveryNEpochs(1);
        }

        public Builder saveEveryNEpochs(int n){
            this.saveEveryNEpochs = n;
            return this;
        }

        public Builder saveEveryNIterations(int n){
            return saveEveryNIterations(n, false);
        }

        public Builder saveEveryNIterations(int n, boolean sinceLast){
            this.saveEveryNIterations = n;
            this.saveEveryNIterSinceLast = sinceLast;
            return this;
        }

        public Builder saveEvery(long amount, TimeUnit timeUnit){
            return saveEvery(amount, timeUnit, false);
        }

        public Builder saveEvery(long amount, TimeUnit timeUnit, boolean sinceLast){
            this.saveEveryAmount = amount;
            this.saveEveryUnit = timeUnit;
            this.saveEverySinceLast = sinceLast;
            return this;
        }



        public Builder keepAll(){
            this.keepMode = KeepMode.ALL;
            return this;
        }

        public Builder keepLast(int n){
            if(n <= 0){
                throw new IllegalArgumentException("Number of model files to keep should be > 0 (got: " + n + ")");
            }
            this.keepMode = KeepMode.LAST;
            this.keepLast = n;
            return this;
        }

        public Builder keepLastAndEvery(int nLast, int everyN){
            if(nLast <= 0){
                throw new IllegalArgumentException("Most recent number of model files to keep should be > 0 (got: "
                        + nLast + ")");
            }
            if(everyN <= 0){
                throw new IllegalArgumentException("Every n model files to keep should be > 0 (got: "
                        + everyN + ")");
            }

            this.keepMode = KeepMode.LAST_AND_EVERY;
            this.keepLast = nLast;
            this.keepEvery = everyN;
            return this;
        }

        public Builder logSaving(boolean logSaving){
            this.logSaving = logSaving;
            return this;
        }

        public CheckpointListener build(){
            return new CheckpointListener(this);
        }

    }

}
