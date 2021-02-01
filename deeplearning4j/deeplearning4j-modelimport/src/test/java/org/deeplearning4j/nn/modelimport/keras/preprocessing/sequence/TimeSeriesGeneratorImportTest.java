/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras.preprocessing.sequence;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.junit.Test;
import org.nd4j.common.resources.Resources;

import java.io.IOException;

/**
 * Import test for Keras TimeSeriesGenerator
 *
 * @author Max Pumperla
 */
public class TimeSeriesGeneratorImportTest extends BaseDL4JTest {

    @Test(timeout=300000)
    public void importTimeSeriesTest() throws IOException, InvalidKerasConfigurationException {
        String path = "modelimport/keras/preprocessing/timeseries_generator.json";

        TimeSeriesGenerator gen = TimeSeriesGenerator.fromJson(Resources.asFile(path).getAbsolutePath());
    }
}
