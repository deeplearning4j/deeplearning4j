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

package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

/**
 * A simple Composite DataSetPreProcessor - allows you to apply multiple DataSetPreProcessors sequentially
 * on the one DataSet, in the order they are passed to the constructor
 *
 * @author Alex Black
 */
public class CompositeDataSetPreProcessor implements DataSetPreProcessor {

    private DataSetPreProcessor[] preProcessors;

    /**
     * @param preProcessors Preprocessors to apply. They will be applied in this order
     */
    public CompositeDataSetPreProcessor(DataSetPreProcessor... preProcessors){
        this.preProcessors = preProcessors;
    }

    @Override
    public void preProcess(DataSet dataSet) {
        for(DataSetPreProcessor p : preProcessors){
            p.preProcess(dataSet);
        }
    }
}
