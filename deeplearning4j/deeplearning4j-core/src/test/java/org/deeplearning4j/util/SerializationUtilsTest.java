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

package org.deeplearning4j.util;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Created by mjk on 9/15/14.
 */
public class SerializationUtilsTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testWriteRead() throws Exception {
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        String irisData = "irisData.dat";

        DataSet freshDataSet = iter.next(150);

        File f = testDir.newFile(irisData);
        org.nd4j.linalg.util.SerializationUtils.saveObject(freshDataSet, f);

        DataSet readDataSet = org.nd4j.linalg.util.SerializationUtils.readObject(f);

        assertEquals(freshDataSet.getFeatures(), readDataSet.getFeatures());
        assertEquals(freshDataSet.getLabels(), readDataSet.getLabels());
    }

}
