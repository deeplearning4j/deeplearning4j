package org.nd4j.autodiff.listeners.checkpoint;


import org.nd4j.shade.guava.io.Files;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.ListenerResponse;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 *
 * CheckpointListener: The goal of this listener is to periodically save a copy of the model during training..<br>
 * Model saving may be done:<br>
 * 1. Every N epochs<br>
 * 2. Every N iterations<br>
 * 3. Every T time units (every 15 minutes, for example)<br>
 * Or some combination of the 3.<br>
 * <br>
 * Models can be restored using {@link #loadCheckpoint(File, int)}, {@link #loadLastCheckpoint(File)} and {@link #loadCheckpoint(int)}.
 * Checkpoints can be obtained using {@link #lastCheckpoint()} and {@link #availableCheckpoints()}
 * <br>
 * <b>Example 1</b>: Saving a checkpoint every 2 epochs, keep all model files
 * <pre>
 * {@code CheckpointListener l = new CheckpointListener.Builder("/save/directory")
 *       .keepAll() //Don't delete any models
 *       .saveEveryNEpochs(2)
 *       .build()
 * }
 * </pre>
 * <br>
 * <b>Example 2</b>: Saving a checkpoint every 1000 iterations, but keeping only the last 3 models (all older model
 * files will be automatically deleted)
 * <pre>
 * {@code CheckpointListener l = new CheckpointListener.Builder(new File("/save/directory"))
 *          .keepLast(3)
 *          .saveEveryNIterations(1000)
 *          .build();
 * }
 * </pre>
 * <br>
 * <b>Example 3</b>: Saving a checkpoint every 15 minutes, keeping the most recent 3 and otherwise every 4th checkpoint
 * file:
 * <pre>
 * {@code CheckpointListener l = new CheckpointListener.Builder(new File("/save/directory"))
 *          .keepLastAndEvery(3, 4)
 *          .saveEvery(15, TimeUnit.MINUTES)
 *          .build();
 * }
 * </pre>
 * <br>
 * Note that you can mix these: for example, to save every epoch and every 15 minutes (independent of last save time):<br>
 * {@code .saveEveryEpoch().saveEvery(15, TimeUnit.MINUTES)}<br>
 * To save every epoch, and every 15 minutes, <i>since the last model save</i> use:<br>
 * {@code .saveEveryEpoch().saveEvery(15, TimeUnit.MINUTES, true)}<br>
 * Note that is this last example, the <i>sinceLast</i> parameter is true. This means the 15-minute counter will be
 * reset any time a model is saved.<br>
 *
 * @author Alex Black
 */
@Slf4j
public class CheckpointListener extends BaseListener implements Serializable {

    private enum KeepMode {ALL, LAST, LAST_AND_EVERY};

    private File rootDir;
    private String fileNamePrefix;
    private KeepMode keepMode;
    private int keepLast;
    private int keepEvery;
    private boolean logSaving;
    private boolean deleteExisting;
    private boolean saveUpdaterState;

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
        this.fileNamePrefix = builder.fileNamePrefix;
        this.keepMode = builder.keepMode;
        this.keepLast = builder.keepLast;
        this.keepEvery = builder.keepEvery;
        this.logSaving = builder.logSaving;
        this.deleteExisting = builder.deleteExisting;
        this.saveUpdaterState = builder.saveUpdaterState;

        this.saveEveryNEpochs = builder.saveEveryNEpochs;
        this.saveEveryNIterations = builder.saveEveryNIterations;
        this.saveEveryNIterSinceLast = builder.saveEveryNIterSinceLast;
        this.saveEveryAmount = builder.saveEveryAmount;
        this.saveEveryUnit = builder.saveEveryUnit;
        this.saveEverySinceLast = builder.saveEverySinceLast;

        if(saveEveryAmount != null){
            saveEveryMs = TimeUnit.MILLISECONDS.convert(saveEveryAmount, saveEveryUnit);
        }

        if(!rootDir.exists()){
            rootDir.mkdir();
        }

        this.checkpointRecordFile = new File(rootDir, "checkpointInfo.txt");
        if(this.checkpointRecordFile.exists() && this.checkpointRecordFile.length() > 0){

            if(deleteExisting){
                //Delete any files matching:
                //"checkpoint_" + checkpointNum + "_" + modelType + ".zip";
                this.checkpointRecordFile.delete();
                File[] files = rootDir.listFiles();
                if(files != null && files.length > 0){
                    for(File f : files){
                        String name = f.getName();
                        if(name.startsWith("checkpoint_") && (name.endsWith("MultiLayerNetwork.zip") || name.endsWith("ComputationGraph.zip"))){
                            f.delete();
                        }
                    }
                }
            } else {
                throw new IllegalStateException("Detected existing checkpoint files at directory " + rootDir.getAbsolutePath() +
                        ". Use deleteExisting(true) to delete existing checkpoint files when present.");
            }
        }
    }

    @Override
    public ListenerResponse epochEnd(SameDiff sameDiff, At at, LossCurve lossCurve, long epochTimeMillis) {
        if(saveEveryNEpochs != null && (at.epoch()+1) % saveEveryNEpochs == 0){
            //Save:
            saveCheckpoint(sameDiff, at);
        }
        //General saving conditions: don't need to check here - will check in iterationDone
        return ListenerResponse.CONTINUE;
    }

    @Override
    public boolean isActive(Operation operation) {
        return operation == Operation.TRAINING;
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        if (startTime < 0) {
            startTime = System.currentTimeMillis();
            startIter = at.iteration();
            return;
        }

        //Check iterations saving condition:
        if(saveEveryNIterations != null){
            if(saveEveryNIterSinceLast){
                //Consider last saved model when deciding whether to save
                long lastSaveIter = (lastCheckpoint != null ? lastCheckpoint.getIteration() : startIter);
                if(at.iteration() - lastSaveIter >= saveEveryNIterations){
                    saveCheckpoint(sd, at);
                    return;
                }
            } else {
                //Same every N iterations, regardless of saving time
                if((at.iteration()+1) % saveEveryNIterations == 0){
                    saveCheckpoint(sd, at);
                    return;
                }
            }
        }

        //Check time saving condition:
        long time = System.currentTimeMillis();
        if(saveEveryUnit != null){
            if(saveEverySinceLast){
                //Consider last saved when deciding whether to save
                long lastSaveTime = (lastCheckpoint != null ? lastCheckpoint.getTimestamp() : startTime);
                if((time - lastSaveTime) >= saveEveryMs){
                    saveCheckpoint(sd, at);
                    return;
                }
            } else {
                //Save periodically, regardless of when last model was saved
                long lastSave = (lastSaveEveryMsNoSinceLast != null ? lastSaveEveryMsNoSinceLast : startTime);
                if((time - lastSave) > saveEveryMs){
                    saveCheckpoint(sd, at);
                    lastSaveEveryMsNoSinceLast = time;
                    return;
                }
            }
        }
    }

    private void saveCheckpoint(SameDiff sd, At at) {
        try{
            saveCheckpointHelper(sd, at);
        } catch (Exception e){
            throw new RuntimeException("Error saving checkpoint", e);
        }
    }

    private void saveCheckpointHelper(SameDiff model, At at) throws Exception {
        if(!checkpointRecordFile.exists()){
            checkpointRecordFile.createNewFile();
            writeCheckpointInfo(Checkpoint.getFileHeader() + "\n", checkpointRecordFile);
        }

        Checkpoint c = new Checkpoint(++lastCheckpointNum, System.currentTimeMillis(), at.iteration(), at.epoch(),null);
        String filename = getFileName(lastCheckpointNum, at, c.getTimestamp());
        c.setFilename(filename);

        File saveFile = new File(rootDir, c.getFilename());
        model.save(saveFile, this.saveUpdaterState);

        String s = c.toFileString();
        writeCheckpointInfo(s + "\n", checkpointRecordFile);

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
                if(cp.getCheckpointNum() > 0 && (cp.getCheckpointNum()+1) % keepEvery == 0){
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

    //Filename format: "<prefix>_checkpoint-#_epoch-#_iter-#_YYYY-MM-dd_HH-MM-ss.bin"
    private String getFileName(int checkpointNum, At at, long time){
        StringBuilder sb = new StringBuilder();
        if(fileNamePrefix != null){
            sb.append(fileNamePrefix);
            if(!fileNamePrefix.endsWith("_")){
                sb.append("_");
            }
        }
        sb.append("checkpoint-")
                .append(checkpointNum)
                .append("_epoch-").append(at.epoch())
                .append("_iter-").append(at.iteration());

        SimpleDateFormat sdf = new SimpleDateFormat("YYYY-MM-dd_HH-mm-ss");
        String date = sdf.format(new Date(time));
        sb.append("_").append(date)
                .append(".bin");

        return sb.toString();
    }

    private static String writeCheckpointInfo(String str, File f){
        try {
            if(!f.exists()){
                f.createNewFile();
            }
            Files.append(str, f, StandardCharsets.UTF_8);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
        return str;
    }

    /**
     * List all available checkpoints. A checkpoint is 'available' if the file can be loaded. Any checkpoint files that
     * have been automatically deleted (given the configuration) will not be returned here.
     *
     * @return List of checkpoint files that can be loaded
     */
    public List<Checkpoint> availableCheckpoints(){
        if(!checkpointRecordFile.exists()){
            return Collections.emptyList();
        }

        return availableCheckpoints(rootDir);
    }

    /**
     * List all available checkpoints. A checkpoint is 'available' if the file can be loaded. Any checkpoint files that
     * have been automatically deleted (given the configuration) will not be returned here.
     * Note that the checkpointInfo.txt file must exist, as this stores checkpoint information
     *
     * @return List of checkpoint files that can be loaded from the specified directory
     */
    public static List<Checkpoint> availableCheckpoints(File directory){
        File checkpointRecordFile = new File(directory, "checkpointInfo.txt");
        Preconditions.checkState(checkpointRecordFile.exists(), "Could not find checkpoint record file at expected path %s", checkpointRecordFile.getAbsolutePath());

        List<String> lines;
        try(InputStream is = new BufferedInputStream(new FileInputStream(checkpointRecordFile))){
            lines = IOUtils.readLines(is);
        } catch (IOException e){
            throw new RuntimeException("Error loading checkpoint data from file: " + checkpointRecordFile.getAbsolutePath(), e);
        }

        List<Checkpoint> out = new ArrayList<>(lines.size()-1); //Assume first line is header
        for( int i=1; i<lines.size(); i++ ){
            Checkpoint c = Checkpoint.fromFileString(lines.get(i));
            if(new File(directory, c.getFilename()).exists()){
                out.add(c);
            }
        }
        return out;
    }

    /**
     * Return the most recent checkpoint, if one exists - otherwise returns null
     * @return Checkpoint
     */
    public Checkpoint lastCheckpoint(){
        if(!checkpointRecordFile.exists()){
            return null;
        }
        return lastCheckpoint(rootDir);
    }

    /**
     * Return the most recent checkpoint, if one exists - otherwise returns null
     * @param rootDir Root direcotry for the checkpoint files
     * @return Checkpoint
     */
    public static Checkpoint lastCheckpoint(File rootDir){
        List<Checkpoint> all = availableCheckpoints(rootDir);
        if(all.isEmpty()){
            return null;
        }
        return all.get(all.size()-1);
    }

    /**
     * Get the model file for the given checkpoint. Checkpoint model file must exist
     *
     * @param checkpoint Checkpoint to get the model file for
     * @return Model file for the checkpoint
     */
    public File getFileForCheckpoint(Checkpoint checkpoint){
        return getFileForCheckpoint(checkpoint.getCheckpointNum());
    }

    /**
     * Get the model file for the given checkpoint number. Checkpoint model file must exist
     *
     * @param checkpointNum Checkpoint number to get the model file for
     * @return Model file for the checkpoint
     */
    public File getFileForCheckpoint(int checkpointNum) {
        return getFileForCheckpoint(rootDir, checkpointNum);
    }

    public static File getFileForCheckpoint(File rootDir, int checkpointNum){
        //Scan the root directory, for a file matching the checkpoint filename pattern:
        //Filename format: "<prefix>_checkpoint-#_epoch-#_iter-#_YYYY-MM-dd_HH-MM-ss.bin"

        if(checkpointNum < 0){
            throw new IllegalArgumentException("Invalid checkpoint number: " + checkpointNum);
        }

        String contains = "_checkpoint-" + checkpointNum + "_epoch-";

        File[] allFiles = rootDir.listFiles();
        if(allFiles != null){
            for(File f : allFiles){
                if(f.getAbsolutePath().contains(contains)){
                    return f;
                }
            }
        }

        throw new IllegalStateException("Model file for checkpoint " + checkpointNum + " does not exist");
    }

    /**
     * Load a given checkpoint number
     *
     * @param loadUpdaterState If true: load the updater state. See {@link SameDiff#load(File, boolean)} for more details
     *
     */
    public SameDiff loadCheckpoint(int checkpointNum, boolean loadUpdaterState){
        return loadCheckpoint(rootDir, checkpointNum, loadUpdaterState);
    }

    /**
     * Load a SameDiff instance for the given checkpoint that resides in the specified root directory
     *
     * @param rootDir          Directory that the checkpoint resides in
     * @param checkpointNum    Checkpoint model number to load
     * @param loadUpdaterState If true: load the updater state. See {@link SameDiff#load(File, boolean)} for more details
     * @return The loaded model
     */
    public static SameDiff loadCheckpoint(File rootDir, int checkpointNum, boolean loadUpdaterState) {
        File f = getFileForCheckpoint(rootDir, checkpointNum);
        return SameDiff.load(f, loadUpdaterState);
    }

    /**
     * Load the last (most recent) checkpoint from the specified root directory
     * @param rootDir Root directory to load checpoint from
     * @return ComputationGraph for last checkpoint
     */
    public static SameDiff loadLastCheckpoint(File rootDir, boolean loadUpdaterState){
        Checkpoint last = lastCheckpoint(rootDir);
        return loadCheckpoint(rootDir, last.getCheckpointNum(), loadUpdaterState);
    }

    public static Builder builder(@NonNull File rootDir){
        return new Builder(rootDir);
    }

    public static class Builder {

        private File rootDir;
        private String fileNamePrefix = "SameDiff";
        private KeepMode keepMode;
        private int keepLast;
        private int keepEvery;
        private boolean saveUpdaterState = true;
        private boolean logSaving = true;
        private boolean deleteExisting = false;

        private Integer saveEveryNEpochs;
        private Integer saveEveryNIterations;
        private boolean saveEveryNIterSinceLast;
        private Long saveEveryAmount;
        private TimeUnit saveEveryUnit;
        private boolean saveEverySinceLast;

        /**
         * @param rootDir Root directory to save models to
         */
        public Builder(@NonNull String rootDir){
            this(new File(rootDir));
        }

        /**
         * @param rootDir Root directory to save models to
         */
        public Builder(@NonNull File rootDir){
            this.rootDir = rootDir;
        }

        public Builder fileNamePrefix(String fileNamePrefix){
            this.fileNamePrefix = fileNamePrefix;
            return this;
        }

        /**
         * Save a model at the end of every epoch
         */
        public Builder saveEveryEpoch(){
            return saveEveryNEpochs(1);
        }

        /**
         * Save a model at the end of every N epochs
         */
        public Builder saveEveryNEpochs(int n){
            this.saveEveryNEpochs = n;
            return this;
        }

        /**
         * Save a model every N iterations
         */
        public Builder saveEveryNIterations(int n){
            return saveEveryNIterations(n, false);
        }

        /**
         * Save a model every N iterations (if sinceLast == false), or if N iterations have passed since
         * the last model vas saved (if sinceLast == true)
         */
        public Builder saveEveryNIterations(int n, boolean sinceLast){
            this.saveEveryNIterations = n;
            this.saveEveryNIterSinceLast = sinceLast;
            return this;
        }

        /**
         * Save a model periodically
         *
         * @param amount   Quantity of the specified time unit
         * @param timeUnit Time unit
         */
        public Builder saveEvery(long amount, TimeUnit timeUnit){
            return saveEvery(amount, timeUnit, false);
        }

        /**
         * Save a model periodically (if sinceLast == false), or if the specified amount of time has elapsed since
         * the last model was saved (if sinceLast == true)
         *
         * @param amount   Quantity of the specified time unit
         * @param timeUnit Time unit
         */
        public Builder saveEvery(long amount, TimeUnit timeUnit, boolean sinceLast){
            this.saveEveryAmount = amount;
            this.saveEveryUnit = timeUnit;
            this.saveEverySinceLast = sinceLast;
            return this;
        }

        /**
         * Keep all model checkpoints - i.e., don't delete any. Note that this is the default.
         */
        public Builder keepAll(){
            this.keepMode = KeepMode.ALL;
            return this;
        }

        /**
         * Keep only the last N most recent model checkpoint files. Older checkpoints will automatically be deleted.
         * @param n Number of most recent checkpoints to keep
         */
        public Builder keepLast(int n){
            if(n <= 0){
                throw new IllegalArgumentException("Number of model files to keep should be > 0 (got: " + n + ")");
            }
            this.keepMode = KeepMode.LAST;
            this.keepLast = n;
            return this;
        }

        /**
         * Keep the last N most recent model checkpoint files, <i>and</i> every M checkpoint files.<br>
         * For example: suppose you save every 100 iterations, for 2050 iteration, and use keepLastAndEvery(3,5).
         * This means after 2050 iterations you would have saved 20 checkpoints - some of which will be deleted.
         * Those remaining in this example: iterations 500, 1000, 1500, 1800, 1900, 2000.
         * @param nLast  Most recent checkpoints to keep
         * @param everyN Every N checkpoints to keep (regardless of age)
         */
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

        /**
         * If true (the default) log a message every time a model is saved
         *
         * @param logSaving Whether checkpoint saves should be logged or not
         */
        public Builder logSaving(boolean logSaving){
            this.logSaving = logSaving;
            return this;
        }

        /**
         * Whether the updater state (history/state for Adam, Nesterov momentum, etc) should be saved with each checkpoint.<br>
         * Updater state is saved by default.
         * If you expect to continue training on any of the checkpoints, this should be set to true. However, it will increase
         * the file size.
         * @param saveUpdaterState If true: updater state will be saved with checkpoints. False: not saved.
         */
        public Builder saveUpdaterState(boolean saveUpdaterState){
            this.saveUpdaterState = saveUpdaterState;
            return this;
        }

        /**
         * If the checkpoint listener is set to save to a non-empty directory, should the CheckpointListener-related
         * content be deleted?<br>
         * This is disabled by default (and instead, an exception will be thrown if existing data is found)<br>
         * WARNING: Be careful when enabling this, as it deletes all saved checkpoint models in the specified directory!
         */
        public Builder deleteExisting(boolean deleteExisting){
            this.deleteExisting = deleteExisting;
            return this;
        }

        public CheckpointListener build(){
            if(saveEveryNEpochs == null && saveEveryAmount == null && saveEveryNIterations == null){
                throw new IllegalStateException("Cannot construct listener: no models will be saved (must use at least" +
                        " one of: save every N epochs, every N iterations, or every T time periods)");
            }

            return new CheckpointListener(this);
        }
    }
}
