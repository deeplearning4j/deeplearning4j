/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.transform.analysis.columns;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;

@Builder
@Data
@AllArgsConstructor
@NoArgsConstructor //For Jackson deserialization
public class StringAnalysis implements ColumnAnalysis {
    private int minLength;
    private int maxLength;
    private double meanLength;
    private double sampleStdevLength;
    private double sampleVarianceLength;
    private long countTotal;
    private double[] histogramBuckets;
    private long[] histogramBucketCounts;

    @Override
    public String toString() {
        return "StringAnalysis(minLen=" + minLength + ",maxLen=" + maxLength + ",meanLen=" + meanLength
                        + ",sampleStDevLen=" + sampleStdevLength + ",sampleVarianceLen=" + sampleVarianceLength
                        + ",count=" + countTotal + ")";
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.String;
    }

}
