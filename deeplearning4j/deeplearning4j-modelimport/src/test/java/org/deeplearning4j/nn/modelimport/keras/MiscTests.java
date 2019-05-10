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

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class MiscTests {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test(timeout = 60000L)
    public void testMultiThreadedLoading() throws Exception {

        String path = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5";
        File root = testDir.newFolder();
        final File f = new ClassPathResource(path).getTempFileFromArchive(root);

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

        String path = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5";
        File root = testDir.newFolder();
        final File f = new ClassPathResource(path).getTempFileFromArchive(root);

        try(InputStream is = new BufferedInputStream(new FileInputStream(f))) {
            MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(is);
            assertNotNull(model);
        }
    }
}
