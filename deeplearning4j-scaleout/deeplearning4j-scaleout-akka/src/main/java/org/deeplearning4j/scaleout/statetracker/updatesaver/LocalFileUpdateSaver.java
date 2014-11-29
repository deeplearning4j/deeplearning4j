package org.deeplearning4j.scaleout.statetracker.updatesaver;

import com.hazelcast.core.EntryEvent;
import com.hazelcast.core.EntryListener;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IMap;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.statetracker.UpdateSaver;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Saves intermittent updates
 * in the directory where the base dir is specified.
 * The default is the tmp directory
 */
public class LocalFileUpdateSaver implements UpdateSaver {

    private Map<String,String> paths;
    private IMap<String,Job> updateableIMap;
    private String baseDir;
    public final static String UPDATE_SAVER = "updatesaver";
    private static Logger log = LoggerFactory.getLogger(LocalFileUpdateSaver.class);


    public LocalFileUpdateSaver(String baseDir,HazelcastInstance instance) {
        this.baseDir = baseDir;
        File dir = new File(baseDir);
        if(!dir.exists())
            dir.mkdirs();
        paths = new ConcurrentHashMap<>();
        if(instance != null) {
            updateableIMap = instance.getMap(UPDATE_SAVER);
            updateableIMap.addEntryListener(new EntryListener<String, Job>() {
                @Override
                public void entryAdded(EntryEvent<String, Job> event) {
                    File saveFile = new File(LocalFileUpdateSaver.this.baseDir,event.getKey());
                    SerializationUtils.saveObject(event.getValue(),saveFile);
                    boolean loadedProperly = false;
                    while(!loadedProperly) {
                        try {
                            SerializationUtils.readObject(saveFile);

                        }catch(Exception e) {

                        }
                        loadedProperly = true;
                    }
                    paths.put(event.getKey(),saveFile.getAbsolutePath());
                    //no longer needed after persistence
                    updateableIMap.remove(event.getKey());
                }

                @Override
                public void entryRemoved(EntryEvent<String, Job> event) {
                }

                @Override
                public void entryUpdated(EntryEvent<String, Job> event) {

                }

                @Override
                public void entryEvicted(EntryEvent<String, Job> event) {
                }
            },true);
        }
       }

    public LocalFileUpdateSaver(String baseDir) {
      this(baseDir,null);
    }

    /**
     * Saves files in the tmp directory
     */
    public LocalFileUpdateSaver() {
        this(System.getProperty("java.io.tmpdir"));
    }

    @Override
    public Job load(String id) throws Exception {
        String path = paths.remove(id);
        if(path == null) {
            log.warn("Tried loading work from id " + id + " but path was null");
            return null;
        }
        File load = new File(path);
        Job u =  SerializationUtils.readObject(load);
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
    public void save(String id,Job save) throws Exception {
        if(save == null)
            throw new IllegalArgumentException("Saving null network not allowed");

        if(updateableIMap != null) {
            updateableIMap.put(id,save);
        }
        else {
            File saveFile = new File(baseDir,id);
            SerializationUtils.saveObject(save,saveFile);
            boolean loadedProperly = false;
            while(!loadedProperly) {
                try {
                    SerializationUtils.readObject(saveFile);

                }catch(Exception e) {

                }
                loadedProperly = true;
            }
            paths.put(id,saveFile.getAbsolutePath());
        }


    }
}
