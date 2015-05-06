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

package org.deeplearning4j.iterativereduce.runtime.irunit;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.deeplearning4j.iterativereduce.runtime.ComputableMaster;
import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;
import org.deeplearning4j.scaleout.api.ir.Updateable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;


/**
 * A very basic way to simulate an IterativeReduce application in order to simply develop/test a new 
 * parallel iterative algorithm. Does not simulate the Avro plumbing.
 *
 */
public class IRUnitDriver<T> {

    private static JobConf defaultConf = new JobConf();
    private static FileSystem localFs = null;
    private static final Logger log = LoggerFactory.getLogger(IRUnitDriver.class);
    static {
        try {
            defaultConf.set("fs.defaultFS", "file:///");
            localFs = FileSystem.getLocal(defaultConf);
        } catch (IOException e) {
            throw new RuntimeException("init failure", e);
        }
    }

    private static Path workDir = new Path("/tmp/");

    Properties props;

    private ComputableMaster master;
    private ArrayList<ComputableWorker> workers;
    private String app_properties_file = "";
    ArrayList<Updateable> worker_results = new ArrayList<>();
    Updateable master_result = null;
    boolean bContinuePass = true;
    String[] props_to_copy = {};
    InputSplit[] splits;

    /**
     * need to load the app.properties file
     *
     * @return
     */
    public Configuration getConfigFromProperties() {

        Configuration c = new Configuration();

        for ( int x = 0; x < props_to_copy.length; x++ ) {
            c.set(props_to_copy[x], this.props.getProperty(props_to_copy[x]));
        }


        return c;

    }

    /**
     * generate splits for this run
     *
     * @param input_path
     * @param job
     * @return
     */
    private InputSplit[] generateDebugSplits(Path input_path, JobConf job) {

        long block_size = localFs.getDefaultBlockSize();

        log.info("default block size: " + (block_size / 1024 / 1024)
                + "MB");

        // ---- set where we'll read the input files from -------------
        FileInputFormat.setInputPaths(job, input_path);

        // try splitting the file in a variety of sizes
        TextInputFormat format = new TextInputFormat();
        format.configure(job);

        int numSplits = 1;

        InputSplit[] splits = null;

        try {
            splits = format.getSplits(job, numSplits);
        } catch (IOException e) {
            log.error("Error with splits",e);
        }

        return splits;

    }

    public IRUnitDriver(String app_prop, String[] props_to_copy) {

        this.app_properties_file = app_prop;
        this.props_to_copy = props_to_copy;

        this.loadPropertiesFile();

    }

    /**
     * Seperated this out from setup() so the tester could set a property before using the properties in setup
     *
     */
    private void loadPropertiesFile() {

        this.props = new Properties();

        try {
            FileInputStream fis = new FileInputStream(this.app_properties_file);
            props.load(fis);
            fis.close();
        } catch (FileNotFoundException ex) {
            log.error("Error with file", ex);
        } catch (IOException ex) {
            log.error("Error with file", ex);
        }


    }

    public void setProperty(String Name, String Value) {

        this.props.setProperty(Name, Value);

    }

    /**
     * setup components of the IR app run 1. load app.properties 2. msg arrays
     * 3. calc local splits 4. setup master 5. setup workers based on number of
     * splits
     *
     */
    public void setup() {
        // setup msg arrays

        // calc splits

        // ---- this all needs to be done in
        JobConf job = new JobConf(defaultConf);

        // app.input.path

        Path splitPath = new Path( props.getProperty("app.input.path") );

        log.info( "app.input.path = " + splitPath );

        // TODO: work on this, splits are generating for everything in dir
        InputSplit[] splits = generateDebugSplits(splitPath, job);

        log.info("split count: " + splits.length);

        try {
            // this.master = (ComputableMaster)
            // custom_master_class.newInstance();

            Class<?> master_clazz = Class.forName(props
                    .getProperty("yarn.master.main"));
            Constructor<?> master_ctor = master_clazz
                    .getConstructor();
            this.master = (ComputableMaster) master_ctor.newInstance(); // new
            log.info("Using master class: " + props
                    .getProperty("yarn.master.main"));

        } catch (Exception e) {
           log.error("Error initializing master",e);
        }

        this.master.setup(this.getConfigFromProperties());

        this.workers = new ArrayList<>();

        log.info("Using worker class: " + props
                .getProperty("yarn.worker.main"));

        for (int x = 0; x < splits.length; x++) {

            ComputableWorker worker = null;
            Class<?> worker_clazz;
            try {
                worker_clazz = Class.forName(props
                        .getProperty("yarn.worker.main"));
                Constructor<?> worker_ctor = worker_clazz
                        .getConstructor();
                worker = (ComputableWorker) worker_ctor.newInstance();

            } catch (Exception e) {
                log.error("Error initializing worker",e);
            }
            // simulates the conf stuff
            worker.setup(this.getConfigFromProperties());



            workers.add(worker);

            log.info("> setup Worker " + x);
        } // for

    }

    /**
     * Simulate a run of the training
     */
    public void simulateRun() {

        List<Updateable> master_results = new ArrayList<>();
        List<Updateable> worker_results = new ArrayList<>();

        long ts_start = System.currentTimeMillis();

        log.info("start-ms:" + ts_start);

        int iterations = Integer.parseInt(props
                .getProperty("app.iteration.count"));

        log.info("Starting Iterations...");

        for (int x = 0; x < iterations; x++) {

            for (int worker_id = 0; worker_id < workers.size(); worker_id++) {

                Updateable result = workers.get(worker_id).compute();
                worker_results.add(result);


            } // for

            Updateable master_result = this.master.compute(worker_results,
                    master_results);


            // process global updates
            for (int worker_id = 0; worker_id < workers.size(); worker_id++) {

                workers.get(worker_id).update(master_result);

            }


        } // for
    
    



    }

    public ComputableMaster getMaster() {

        return this.master;

    }

}