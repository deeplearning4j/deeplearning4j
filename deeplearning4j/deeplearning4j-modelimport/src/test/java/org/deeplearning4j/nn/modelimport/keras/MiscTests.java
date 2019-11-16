/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.modelimport.keras;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.DL4JKerasModelValidator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.resources.Resources;
import org.nd4j.validation.ValidationResult;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class MiscTests extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test(timeout = 60000L)
    public void testMultiThreadedLoading() throws Exception {
        final File f = Resources.asFile("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5");

        int numThreads = 4;
        final CountDownLatch latch = new CountDownLatch(numThreads);
        final AtomicInteger errors = new AtomicInteger();
        for( int i=0; i<numThreads; i++ ){
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        for (int i = 0; i < 20; i++) {
                            //System.out.println("Iteration " + i + ": " + Thread.currentThread().getId());
                            try {
                                //System.out.println("About to load: " + Thread.currentThread().getId());
                                KerasSequentialModel kerasModel = new KerasModel().modelBuilder().modelHdf5Filename(f.getAbsolutePath())
                                        .enforceTrainingConfig(false).buildSequential();
                                //System.out.println("Loaded Keras: " + Thread.currentThread().getId());

                                MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();
                                Thread.sleep(50);
                            } catch (Throwable t) {
                                t.printStackTrace();
                                errors.getAndIncrement();
                            }

                        }
                    } catch (Throwable t){
                        t.printStackTrace();
                        errors.getAndIncrement();
                    } finally {
                        latch.countDown();
                    }
                }
            }).start();
        }

        boolean result = latch.await(30000, TimeUnit.MILLISECONDS);
        assertTrue("Latch did not get to 0", result);
        assertEquals("Number of errors", 0, errors.get());
    }

    @Test(timeout = 60000L)
    public void testLoadFromStream() throws Exception {
        final File f = Resources.asFile("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5");

        try(InputStream is = new BufferedInputStream(new FileInputStream(f))) {
            MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(is);
            assertNotNull(model);
        }
    }

    @Test(timeout = 60000L)
    public void testModelValidatorSequential() throws Exception {
        File f = testDir.newFolder();

        //Test not existent file:
        File fNonExistent = new File("doesntExist.h5");
        ValidationResult vr0 = DL4JKerasModelValidator.validateKerasSequential(fNonExistent);
        assertFalse(vr0.isValid());
        assertEquals("Keras Sequential Model HDF5", vr0.getFormatType());
        assertTrue(vr0.getIssues().get(0), vr0.getIssues().get(0).contains("exist"));
        System.out.println(vr0.toString());

        //Test empty file:
        File fEmpty = new File(f, "empty.h5");
        fEmpty.createNewFile();
        assertTrue(fEmpty.exists());
        ValidationResult vr1 = DL4JKerasModelValidator.validateKerasSequential(fEmpty);
        assertEquals("Keras Sequential Model HDF5", vr1.getFormatType());
        assertFalse(vr1.isValid());
        assertTrue(vr1.getIssues().get(0), vr1.getIssues().get(0).contains("empty"));
        System.out.println(vr1.toString());

        //Test directory (not zip file)
        File directory = new File(f, "dir");
        boolean created = directory.mkdir();
        assertTrue(created);
        ValidationResult vr2 = DL4JKerasModelValidator.validateKerasSequential(directory);
        assertEquals("Keras Sequential Model HDF5", vr2.getFormatType());
        assertFalse(vr2.isValid());
        assertTrue(vr2.getIssues().get(0), vr2.getIssues().get(0).contains("directory"));
        System.out.println(vr2.toString());

        //Test Keras HDF5 format:
        File fText = new File(f, "text.txt");
        FileUtils.writeStringToFile(fText, "Not a hdf5 file :)", StandardCharsets.UTF_8);
        ValidationResult vr3 = DL4JKerasModelValidator.validateKerasSequential(fText);
        assertEquals("Keras Sequential Model HDF5", vr3.getFormatType());
        assertFalse(vr3.isValid());
        String s = vr3.getIssues().get(0);
        assertTrue(s, s.contains("Keras") && s.contains("Sequential") && s.contains("corrupt"));
        System.out.println(vr3.toString());

        //Test corrupted npy format:
        File fValid = Resources.asFile("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5");
        byte[] numpyBytes = FileUtils.readFileToByteArray(fValid);
        for( int i=0; i<30; i++ ){
            numpyBytes[i] = 0;
        }
        File fCorrupt = new File(f, "corrupt.h5");
        FileUtils.writeByteArrayToFile(fCorrupt, numpyBytes);

        ValidationResult vr4 = DL4JKerasModelValidator.validateKerasSequential(fCorrupt);
        assertEquals("Keras Sequential Model HDF5", vr4.getFormatType());
        assertFalse(vr4.isValid());
        s = vr4.getIssues().get(0);
        assertTrue(s, s.contains("Keras") && s.contains("Sequential") && s.contains("corrupt"));
        System.out.println(vr4.toString());


        //Test valid npy format:
        ValidationResult vr5 = DL4JKerasModelValidator.validateKerasSequential(fValid);
        assertEquals("Keras Sequential Model HDF5", vr5.getFormatType());
        assertTrue(vr5.isValid());
        assertNull(vr5.getIssues());
        assertNull(vr5.getException());
        System.out.println(vr4.toString());
    }

    @Test(timeout = 60000L)
    public void testModelValidatorFunctional() throws Exception {
        File f = testDir.newFolder();
        //String modelPath = "modelimport/keras/examples/functional_lstm/lstm_functional_tf_keras_2.h5";

        //Test not existent file:
        File fNonExistent = new File("doesntExist.h5");
        ValidationResult vr0 = DL4JKerasModelValidator.validateKerasFunctional(fNonExistent);
        assertFalse(vr0.isValid());
        assertEquals("Keras Functional Model HDF5", vr0.getFormatType());
        assertTrue(vr0.getIssues().get(0), vr0.getIssues().get(0).contains("exist"));
        System.out.println(vr0.toString());

        //Test empty file:
        File fEmpty = new File(f, "empty.h5");
        fEmpty.createNewFile();
        assertTrue(fEmpty.exists());
        ValidationResult vr1 = DL4JKerasModelValidator.validateKerasFunctional(fEmpty);
        assertEquals("Keras Functional Model HDF5", vr1.getFormatType());
        assertFalse(vr1.isValid());
        assertTrue(vr1.getIssues().get(0), vr1.getIssues().get(0).contains("empty"));
        System.out.println(vr1.toString());

        //Test directory (not zip file)
        File directory = new File(f, "dir");
        boolean created = directory.mkdir();
        assertTrue(created);
        ValidationResult vr2 = DL4JKerasModelValidator.validateKerasFunctional(directory);
        assertEquals("Keras Functional Model HDF5", vr2.getFormatType());
        assertFalse(vr2.isValid());
        assertTrue(vr2.getIssues().get(0), vr2.getIssues().get(0).contains("directory"));
        System.out.println(vr2.toString());

        //Test Keras HDF5 format:
        File fText = new File(f, "text.txt");
        FileUtils.writeStringToFile(fText, "Not a hdf5 file :)", StandardCharsets.UTF_8);
        ValidationResult vr3 = DL4JKerasModelValidator.validateKerasFunctional(fText);
        assertEquals("Keras Functional Model HDF5", vr3.getFormatType());
        assertFalse(vr3.isValid());
        String s = vr3.getIssues().get(0);
        assertTrue(s, s.contains("Keras") && s.contains("Functional") && s.contains("corrupt"));
        System.out.println(vr3.toString());

        //Test corrupted npy format:
        File fValid = Resources.asFile("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5");
        byte[] numpyBytes = FileUtils.readFileToByteArray(fValid);
        for( int i=0; i<30; i++ ){
            numpyBytes[i] = 0;
        }
        File fCorrupt = new File(f, "corrupt.h5");
        FileUtils.writeByteArrayToFile(fCorrupt, numpyBytes);

        ValidationResult vr4 = DL4JKerasModelValidator.validateKerasFunctional(fCorrupt);
        assertEquals("Keras Functional Model HDF5", vr4.getFormatType());
        assertFalse(vr4.isValid());
        s = vr4.getIssues().get(0);
        assertTrue(s, s.contains("Keras") && s.contains("Functional") && s.contains("corrupt"));
        System.out.println(vr4.toString());


        //Test valid npy format:
        ValidationResult vr5 = DL4JKerasModelValidator.validateKerasFunctional(fValid);
        assertEquals("Keras Functional Model HDF5", vr5.getFormatType());
        assertTrue(vr5.isValid());
        assertNull(vr5.getIssues());
        assertNull(vr5.getException());
        System.out.println(vr4.toString());
    }
}
