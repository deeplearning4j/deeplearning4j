/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.observation.preprocessors;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * The PermuteDataSetPreProcessor will change the dataset from NCHW to NHWC or NHWC to NCHW.
 *
 * @author Alexandre Boulanger
 */
public class PermuteDataSetPreProcessor implements DataSetPreProcessor {

    private final PermutationTypes permutationType;

    public enum PermutationTypes { NCHWtoNHWC, NHWCtoNCHW }

    public PermuteDataSetPreProcessor(PermutationTypes permutationType) {

        this.permutationType = permutationType;
    }

    @Override
    public void preProcess(DataSet dataSet) {
        Preconditions.checkNotNull(dataSet, "Encountered null dataSet");

        if(dataSet.isEmpty()) {
            return;
        }

        INDArray input = dataSet.getFeatures();
        INDArray output;
        switch (permutationType) {
            case NCHWtoNHWC:
                output = input.permute(0, 2, 3, 1);
                break;

            case NHWCtoNCHW:
                output = input.permute(0, 3, 1, 2);
                break;

            default:
                output = input;
                break;
        }

        dataSet.setFeatures(output);
    }
}
