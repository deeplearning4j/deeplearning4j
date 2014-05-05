package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;

import org.deeplearning4j.iterativereduce.tracker.statetracker.UpdateSaver;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.util.SerializationUtils;

import java.io.File;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Saves intermittent updates in the directory where the base dir is specified.
 * The default is the tmp directory
 */
public class LocalFileUpdateSaver implements UpdateSaver<UpdateableImpl> {

    private Map<String,String> paths = new ConcurrentHashMap<>();
    private String baseDir;

    public LocalFileUpdateSaver(String baseDir) {
        this.baseDir = baseDir;
    }

    /**
     * Saves files in the tmp directory
     */
    public LocalFileUpdateSaver() {
        this(System.getProperty("java.io.tmpdir"));
    }

    @Override
    public UpdateableImpl load(String id) throws Exception {
        String path = paths.remove(id);
        File load = new File(path);
        UpdateableImpl u =  SerializationUtils.readObject(load);
        load.delete();
        return u;
    }

    /**
     * Cleans up the persistence layer.
     * This will usually be used to clear up left over files from updates
     */
    @Override
    public void cleanup() {
        for(String s : paths.values())
            new File(s).delete();
    }

    @Override
    public void save(String id,UpdateableImpl save) throws Exception {
       if(save.get() == null)
           throw new IllegalArgumentException("Saving null network not allowed");
        File saveFile = new File(id);
        SerializationUtils.saveObject(save,saveFile);
        boolean loadedProperly = false;
        while(!loadedProperly) {
            try {
                UpdateableImpl u =  SerializationUtils.readObject(saveFile);

            }catch(Exception e) {

            }
            loadedProperly = true;
        }
        paths.put(id,saveFile.getAbsolutePath());

    }
}
