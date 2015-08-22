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
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
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
}
