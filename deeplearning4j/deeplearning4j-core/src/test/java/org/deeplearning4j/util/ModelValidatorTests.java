/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.util;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.common.validation.ValidationResult;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.*;

public class ModelValidatorTests extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testMultiLayerNetworkValidation() throws Exception {
        File f = testDir.newFolder();

        //Test non-existent file
        File f0 = new File(f, "doesntExist.bin");
        ValidationResult vr0 = DL4JModelValidator.validateMultiLayerNetwork(f0);
        assertFalse(vr0.isValid());
        assertTrue(vr0.getIssues().get(0).contains("exist"));
        assertEquals("MultiLayerNetwork", vr0.getFormatType());
        assertEquals(MultiLayerNetwork.class, vr0.getFormatClass());
        assertNull(vr0.getException());
//        System.out.println(vr0.toString());

        //Test empty file
        File f1 = new File(f, "empty.bin");
        f1.createNewFile();
        assertTrue(f1.exists());
        ValidationResult vr1 = DL4JModelValidator.validateMultiLayerNetwork(f1);
        assertFalse(vr1.isValid());
        assertTrue(vr1.getIssues().get(0).contains("empty"));
        assertEquals("MultiLayerNetwork", vr1.getFormatType());
        assertEquals(MultiLayerNetwork.class, vr1.getFormatClass());
        assertNull(vr1.getException());
//        System.out.println(vr1.toString());

        //Test invalid zip file
        File f2 = new File(f, "notReallyZip.zip");
        FileUtils.writeStringToFile(f2, "This isn't actually a zip file", StandardCharsets.UTF_8);
        ValidationResult vr2 = DL4JModelValidator.validateMultiLayerNetwork(f2);
        assertFalse(vr2.isValid());
        String s = vr2.getIssues().get(0);
        assertTrue(s, s.contains("zip") && s.contains("corrupt"));
        assertEquals("MultiLayerNetwork", vr2.getFormatType());
        assertEquals(MultiLayerNetwork.class, vr2.getFormatClass());
        assertNotNull(vr2.getException());
//        System.out.println(vr2.toString());

        //Test valid zip, but missing configuration
        File f3 = new File(f, "modelNoConfig.zip");
        getSimpleNet().save(f3);
        try (FileSystem zipfs = FileSystems.newFileSystem(URI.create("jar:" + f3.toURI().toString()), Collections.singletonMap("create", "false"))) {
            Path p = zipfs.getPath(ModelSerializer.CONFIGURATION_JSON);
            Files.delete(p);
        }
        ValidationResult vr3 = DL4JModelValidator.validateMultiLayerNetwork(f3);
        assertFalse(vr3.isValid());
        s = vr3.getIssues().get(0);
        assertEquals(1, vr3.getIssues().size());
        assertTrue(s, s.contains("missing") && s.contains("configuration"));
        assertEquals("MultiLayerNetwork", vr3.getFormatType());
        assertEquals(MultiLayerNetwork.class, vr3.getFormatClass());
        assertNull(vr3.getException());
//        System.out.println(vr3.toString());


        //Test valid sip, but missing params
        File f4 = new File(f, "modelNoParams.zip");
        getSimpleNet().save(f4);
        try (FileSystem zipfs = FileSystems.newFileSystem(URI.create("jar:" + f4.toURI().toString()), Collections.singletonMap("create", "false"))) {
            Path p = zipfs.getPath(ModelSerializer.COEFFICIENTS_BIN);
            Files.delete(p);
        }
        ValidationResult vr4 = DL4JModelValidator.validateMultiLayerNetwork(f4);
        assertFalse(vr4.isValid());
        s = vr4.getIssues().get(0);
        assertEquals(1, vr4.getIssues().size());
        assertTrue(s, s.contains("missing") && s.contains("coefficients"));
        assertEquals("MultiLayerNetwork", vr4.getFormatType());
        assertEquals(MultiLayerNetwork.class, vr4.getFormatClass());
        assertNull(vr4.getException());
//        System.out.println(vr4.toString());


        //Test valid model
        File f5 = new File(f, "modelValid.zip");
        getSimpleNet().save(f5);
        ValidationResult vr5 = DL4JModelValidator.validateMultiLayerNetwork(f5);
        assertTrue(vr5.isValid());
        assertNull(vr5.getIssues());
        assertEquals("MultiLayerNetwork", vr5.getFormatType());
        assertEquals(MultiLayerNetwork.class, vr5.getFormatClass());
        assertNull(vr5.getException());
//        System.out.println(vr5.toString());


        //Test valid model with corrupted JSON
        File f6 = new File(f, "modelBadJson.zip");
        getSimpleNet().save(f6);
        try(ZipFile zf = new ZipFile(f5); ZipOutputStream zo = new ZipOutputStream(new BufferedOutputStream(new FileOutputStream(f6)))){
            Enumeration<? extends ZipEntry> e = zf.entries();
            while(e.hasMoreElements()){
                ZipEntry ze = e.nextElement();
                zo.putNextEntry(new ZipEntry(ze.getName()));
                if(ze.getName().equals(ModelSerializer.CONFIGURATION_JSON)){
                    zo.write("totally not valid json! - {}".getBytes(StandardCharsets.UTF_8));
                } else {
                    byte[] bytes;
                    try(ZipInputStream zis = new ZipInputStream(zf.getInputStream(ze))){
                        bytes = IOUtils.toByteArray(zis);
                    }
                    zo.write(bytes);
//                    System.out.println("WROTE: " + ze.getName());
                }
            }
        }
        ValidationResult vr6 = DL4JModelValidator.validateMultiLayerNetwork(f6);
        assertFalse(vr6.isValid());
        s = vr6.getIssues().get(0);
        assertEquals(1, vr6.getIssues().size());
        assertTrue(s, s.contains("JSON") && s.contains("valid") && s.contains("MultiLayerConfiguration"));
        assertEquals("MultiLayerNetwork", vr6.getFormatType());
        assertEquals(MultiLayerNetwork.class, vr6.getFormatClass());
        assertNotNull(vr6.getException());
//        System.out.println(vr6.toString());
    }


    @Test
    public void testComputationGraphNetworkValidation() throws Exception {
        File f = testDir.newFolder();

        //Test non-existent file
        File f0 = new File(f, "doesntExist.bin");
        ValidationResult vr0 = DL4JModelValidator.validateComputationGraph(f0);
        assertFalse(vr0.isValid());
        assertTrue(vr0.getIssues().get(0).contains("exist"));
        assertEquals("ComputationGraph", vr0.getFormatType());
        assertEquals(ComputationGraph.class, vr0.getFormatClass());
        assertNull(vr0.getException());
//        System.out.println(vr0.toString());

        //Test empty file
        File f1 = new File(f, "empty.bin");
        f1.createNewFile();
        assertTrue(f1.exists());
        ValidationResult vr1 = DL4JModelValidator.validateComputationGraph(f1);
        assertFalse(vr1.isValid());
        assertTrue(vr1.getIssues().get(0).contains("empty"));
        assertEquals("ComputationGraph", vr1.getFormatType());
        assertEquals(ComputationGraph.class, vr1.getFormatClass());
        assertNull(vr1.getException());
//        System.out.println(vr1.toString());

        //Test invalid zip file
        File f2 = new File(f, "notReallyZip.zip");
        FileUtils.writeStringToFile(f2, "This isn't actually a zip file", StandardCharsets.UTF_8);
        ValidationResult vr2 = DL4JModelValidator.validateComputationGraph(f2);
        assertFalse(vr2.isValid());
        String s = vr2.getIssues().get(0);
        assertTrue(s, s.contains("zip") && s.contains("corrupt"));
        assertEquals("ComputationGraph", vr2.getFormatType());
        assertEquals(ComputationGraph.class, vr2.getFormatClass());
        assertNotNull(vr2.getException());
//        System.out.println(vr2.toString());

        //Test valid zip, but missing configuration
        File f3 = new File(f, "modelNoConfig.zip");
        getSimpleNet().save(f3);
        try (FileSystem zipfs = FileSystems.newFileSystem(URI.create("jar:" + f3.toURI().toString()), Collections.singletonMap("create", "false"))) {
            Path p = zipfs.getPath(ModelSerializer.CONFIGURATION_JSON);
            Files.delete(p);
        }
        ValidationResult vr3 = DL4JModelValidator.validateComputationGraph(f3);
        assertFalse(vr3.isValid());
        s = vr3.getIssues().get(0);
        assertEquals(1, vr3.getIssues().size());
        assertTrue(s, s.contains("missing") && s.contains("configuration"));
        assertEquals("ComputationGraph", vr3.getFormatType());
        assertEquals(ComputationGraph.class, vr3.getFormatClass());
        assertNull(vr3.getException());
//        System.out.println(vr3.toString());


        //Test valid sip, but missing params
        File f4 = new File(f, "modelNoParams.zip");
        getSimpleNet().save(f4);
        try (FileSystem zipfs = FileSystems.newFileSystem(URI.create("jar:" + f4.toURI().toString()), Collections.singletonMap("create", "false"))) {
            Path p = zipfs.getPath(ModelSerializer.COEFFICIENTS_BIN);
            Files.delete(p);
        }
        ValidationResult vr4 = DL4JModelValidator.validateComputationGraph(f4);
        assertFalse(vr4.isValid());
        s = vr4.getIssues().get(0);
        assertEquals(1, vr4.getIssues().size());
        assertTrue(s, s.contains("missing") && s.contains("coefficients"));
        assertEquals("ComputationGraph", vr4.getFormatType());
        assertEquals(ComputationGraph.class, vr4.getFormatClass());
        assertNull(vr4.getException());
//        System.out.println(vr4.toString());


        //Test valid model
        File f5 = new File(f, "modelValid.zip");
        getSimpleNet().save(f5);
        ValidationResult vr5 = DL4JModelValidator.validateComputationGraph(f5);
        assertTrue(vr5.isValid());
        assertNull(vr5.getIssues());
        assertEquals("ComputationGraph", vr5.getFormatType());
        assertEquals(ComputationGraph.class, vr5.getFormatClass());
        assertNull(vr5.getException());
//        System.out.println(vr5.toString());


        //Test valid model with corrupted JSON
        File f6 = new File(f, "modelBadJson.zip");
        getSimpleNet().save(f6);
        try(ZipFile zf = new ZipFile(f5); ZipOutputStream zo = new ZipOutputStream(new BufferedOutputStream(new FileOutputStream(f6)))){
            Enumeration<? extends ZipEntry> e = zf.entries();
            while(e.hasMoreElements()){
                ZipEntry ze = e.nextElement();
                zo.putNextEntry(new ZipEntry(ze.getName()));
                if(ze.getName().equals(ModelSerializer.CONFIGURATION_JSON)){
                    zo.write("totally not valid json! - {}".getBytes(StandardCharsets.UTF_8));
                } else {
                    byte[] bytes;
                    try(ZipInputStream zis = new ZipInputStream(zf.getInputStream(ze))){
                        bytes = IOUtils.toByteArray(zis);
                    }
                    zo.write(bytes);
//                    System.out.println("WROTE: " + ze.getName());
                }
            }
        }
        ValidationResult vr6 = DL4JModelValidator.validateComputationGraph(f6);
        assertFalse(vr6.isValid());
        s = vr6.getIssues().get(0);
        assertEquals(1, vr6.getIssues().size());
        assertTrue(s, s.contains("JSON") && s.contains("valid") && s.contains("ComputationGraphConfiguration"));
        assertEquals("ComputationGraph", vr6.getFormatType());
        assertEquals(ComputationGraph.class, vr6.getFormatClass());
        assertNotNull(vr6.getException());
//        System.out.println(vr6.toString());
    }



    public static MultiLayerNetwork getSimpleNet(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    public static ComputationGraph getSimpleCG(){
        return getSimpleNet().toComputationGraph();
    }
}
