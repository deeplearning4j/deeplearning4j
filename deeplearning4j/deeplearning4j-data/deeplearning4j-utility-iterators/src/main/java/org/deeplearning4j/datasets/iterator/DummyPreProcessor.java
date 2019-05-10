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

package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * This is special dummy preProcessor, that does nothing.
 *
 * @author raver119@gmail.com
 */
public class DummyPreProcessor implements DataSetPreProcessor {
    /**
     * Pre process a dataset
     *
     * @param toPreProcess the data set to pre process
     */
    @Override
    public void preProcess(DataSet toPreProcess) {
        // no-op
    }
}
