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

package org.deeplearning4j.rl4j.observation.preprocessor;

import lombok.Builder;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * The SkippingDataSetPreProcessor will either do nothing to the input (when not skipped) or will empty
 * the input DataSet when skipping.
 *
 * @author Alexandre Boulanger
 */
public class SkippingDataSetPreProcessor extends ResettableDataSetPreProcessor {

    private final int skipFrame;

    private int currentIdx = 0;

    /**
     * @param skipFrame For example, a skipFrame of 4 will skip 3 out of 4 observations.
     */
    @Builder
    public SkippingDataSetPreProcessor(int skipFrame) {
        Preconditions.checkArgument(skipFrame > 0, "skipFrame must be greater than 0, got %s", skipFrame);
        this.skipFrame = skipFrame;
    }

    @Override
    public void preProcess(DataSet dataSet) {
        Preconditions.checkNotNull(dataSet, "Encountered null dataSet");

        if(dataSet.isEmpty()) {
            return;
        }

        if(currentIdx++ % skipFrame != 0) {
            dataSet.setFeatures(null);
            dataSet.setLabels(null);
        }
    }

    @Override
    public void reset() {
        currentIdx = 0;
    }
}
