/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.scaleout.statetracker.updatesaver;

import com.hazelcast.core.*;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.api.statetracker.UpdateSaver;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;
import java.util.UUID;
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
    private static final Logger log = LoggerFactory.getLogger(LocalFileUpdateSaver.class);


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
                    String fileName = event.getKey();
                    if(event.getKey().equals("."))
                        fileName = UUID.randomUUID().toString();
                    File saveFile = new File(LocalFileUpdateSaver.this.baseDir,fileName);
                    if(saveFile.isDirectory()) {
                        saveFile = new File(LocalFileUpdateSaver.this.baseDir,UUID.randomUUID().toString());
                    }
                    SerializationUtils.saveObject(event.getValue(),saveFile);

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

                @Override
                public void mapEvicted(MapEvent mapEvent) {

                }

                @Override
                public void mapCleared(MapEvent mapEvent) {

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
    public synchronized Job load(String id) throws Exception {
        String path = paths.remove(id);
        if(path == null) {
            log.warn("Tried loading work from id " + id + " but path was null");
            return null;
        }
        File load = new File(path);
        Job u =  SerializationUtils.readObject(load);
        load.deleteOnExit();
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
