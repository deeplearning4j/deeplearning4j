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

package org.datavec.api.transform.analysis.sequence;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

/**
 * Created by Alex on 12/03/2016.
 */
@AllArgsConstructor
@Data
@Builder
public class SequenceLengthAnalysis implements Serializable {

    private long totalNumSequences;
    private int minSeqLength;
    private int maxSeqLength;
    private long countZeroLength;
    private long countOneLength;
    private double meanLength;
    private double[] histogramBuckets;
    private long[] histogramBucketCounts;

    protected SequenceLengthAnalysis(){
        //No-arg for JSON
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("SequenceLengthAnalysis(").append("totalNumSequences=").append(totalNumSequences)
                        .append(",minSeqLength=").append(minSeqLength).append(",maxSeqLength=").append(maxSeqLength)
                        .append(",countZeroLength=").append(countZeroLength).append(",countOneLength=")
                        .append(countOneLength).append(",meanLength=").append(meanLength).append(")");
        return sb.toString();
    }

}
