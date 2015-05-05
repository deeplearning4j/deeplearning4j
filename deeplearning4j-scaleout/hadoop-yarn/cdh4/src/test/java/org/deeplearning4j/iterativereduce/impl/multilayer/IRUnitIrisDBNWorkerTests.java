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

package org.deeplearning4j.iterativereduce.impl.multilayer;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.deeplearning4j.iterativereduce.irunit.IRUnitDriver;
import org.deeplearning4j.nn.layers.feedforward.rbm.RBM;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;


public class IRUnitIrisDBNWorkerTests {

    private static JobConf defaultConf = new JobConf();
    private static FileSystem localFs = null;
    static {
        try {
            defaultConf.set("fs.defaultFS", "file:///");
            localFs = FileSystem.getLocal(defaultConf);
        } catch (IOException e) {
            throw new RuntimeException("init failure", e);
        }
    }

    private InputSplit[] generateDebugSplits(Path input_path, JobConf job) {

        long block_size = localFs.getDefaultBlockSize();

        System.out.println("default block size: " + (block_size / 1024 / 1024)
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
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return splits;

    }

    @Test
    public void before() {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

    }

    @Test
    public void testSingleWorkerConfigSetup() {

    }

    @Test
    public void testSingleWorker() throws Exception {
        IRUnitDriver polr_ir = new IRUnitDriver(new ClassPathResource("yarn/configurations/svmLightWorkerIRUnitTest.properties").getFile().getAbsolutePath());
        polr_ir.setup();
        polr_ir.simulateRun();


    }

    @Test
    public void testTwoWorkers()  throws Exception {

        IRUnitDriver polr_ir = new IRUnitDriver(new ClassPathResource("/yarn/configurations/svmLightIris_TwoWorkers_IRUnitTest.properties").getFile().getAbsolutePath());
        polr_ir.setup();
        polr_ir.simulateRun();


    }

    @Test
    public void testThreeWorkers()  throws Exception {

        IRUnitDriver polr_ir = new IRUnitDriver(new ClassPathResource("/yarn/configurations/svmLightIris_ThreeWorkers_IRUnitTest.properties").getFile().getAbsolutePath());
        polr_ir.setup();
        polr_ir.simulateRun();


    }


    @Test
    public void testConfIssues() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().layerFactory(LayerFactories.getFactory(RBM.class))
                .list(3).hiddenLayerSizes(new int[]{3,2}).build();
        String json = conf.toJson();

        Configuration c = new Configuration();
        c.set( "test_json", json );

        String key = c.get("test_json");
        assertEquals(json,key);

        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson( key );
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf2);
        assertEquals(3, multiLayerNetwork.getnLayers());
        assertArrayEquals(new int[]{3,2}, multiLayerNetwork.getLayerWiseConfigurations().getHiddenLayerSizes());
    }

    @Test
    public void loadFromFileProps() throws Exception {

        String props_file = new ClassPathResource("/yarn/configurations/svmLightWorkerIRUnitTest.properties").getFile().getAbsolutePath();

        Properties props = new Properties();

        try {
            FileInputStream fis = new FileInputStream( props_file );
            props.load(fis);
            fis.close();
        } catch (IOException ex) {
            // throw ex; // TODO: be nice
            System.out.println(ex);
        }

        String json = props.getProperty("org.deeplearning4j.scaleout.multilayerconf");

        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson( json );
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf2);
        assertArrayEquals(new int[]{2,2}, multiLayerNetwork.getLayerWiseConfigurations().getHiddenLayerSizes());

    }

}
