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

package org.deeplearning4j.perf.listener;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

import static org.junit.Assert.assertEquals;

public class TestSystemInfoPrintListener {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testListener() throws Exception {
        SystemInfoPrintListener systemInfoPrintListener = SystemInfoPrintListener.builder()
                .printOnEpochStart(true).printOnEpochEnd(true)
                .build();

        File tmpFile = testDir.newFile("tmpfile-log.txt");
        assertEquals(0, tmpFile.length() );

        SystemInfoFilePrintListener systemInfoFilePrintListener = SystemInfoFilePrintListener.builder()
                .printOnEpochStart(true).printOnEpochEnd(true).printFileTarget(tmpFile)
                .build();
        tmpFile.deleteOnExit();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new OutputLayer.Builder().nIn(4).nOut(3).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(systemInfoFilePrintListener);

        DataSetIterator iter = new IrisDataSetIterator(10, 150);

        net.fit(iter, 3);

//        System.out.println(FileUtils.readFileToString(tmpFile));
    }

}
