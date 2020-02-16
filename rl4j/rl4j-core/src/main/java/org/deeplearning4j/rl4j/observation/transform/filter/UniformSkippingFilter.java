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
package org.deeplearning4j.rl4j.observation.transform.filter;

import org.deeplearning4j.rl4j.observation.transform.FilterOperation;
import org.nd4j.base.Preconditions;
import java.util.Map;

/**
 * Used with {@link org.deeplearning4j.rl4j.observation.transform.TransformProcess TransformProcess}. Will cause the
 * transform process to skip a fixed number of frames between non skipped ones.
 *
 * @author Alexandre Boulanger
 */
public class UniformSkippingFilter implements FilterOperation {

    private final int skipFrame;

    /**
     * @param skipFrame Will cause the filter to keep (not skip) 1 frame every skipFrames.
     */
    public UniformSkippingFilter(int skipFrame) {
        Preconditions.checkArgument(skipFrame > 0, "skipFrame should be greater than 0");

        this.skipFrame = skipFrame;
    }

    @Override
    public boolean isSkipped(Map<String, Object> channelsData, int currentObservationStep, boolean isFinalObservation) {
        return !isFinalObservation && (currentObservationStep % skipFrame != 0);
    }
}
