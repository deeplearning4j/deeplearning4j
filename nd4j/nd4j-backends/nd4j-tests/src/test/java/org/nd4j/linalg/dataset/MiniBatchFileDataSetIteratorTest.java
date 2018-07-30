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

package org.nd4j.linalg.dataset;

import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.IOException;

import static org.junit.Assert.assertEquals;


/**
 * Created by agibsonccc on 9/10/15.
 */
@RunWith(Parameterized.class)
public class MiniBatchFileDataSetIteratorTest extends BaseNd4jTest {

    public MiniBatchFileDataSetIteratorTest(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testMiniBatches() throws Exception {
        DataSet load = new IrisDataSetIterator(150, 150).next();
        final MiniBatchFileDataSetIterator iter = new MiniBatchFileDataSetIterator(load, 10, false);
        while (iter.hasNext())
            assertEquals(10, iter.next().numExamples());
        if (iter.getRootDir() == null)
            return;
        DataSetIterator existing = new ExistingMiniBatchDataSetIterator(iter.getRootDir());
        while (iter.hasNext())
            assertEquals(10, existing.next().numExamples());


        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    FileUtils.deleteDirectory(iter.getRootDir());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }));

    }

    @Override
    public char ordering() {
        return 'f';
    }
}

