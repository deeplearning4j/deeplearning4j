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

package org.datavec.api.transform.analysis.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;

/**
 * Analysis for Long columns
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor //For Jackson deserialization
public class LongAnalysis extends NumericalColumnAnalysis {

    private long min;
    private long max;

    private LongAnalysis(Builder builder) {
        super(builder);
        this.min = builder.min;
        this.max = builder.max;
    }

    @Override
    public String toString() {
        return "LongAnalysis(min=" + min + ",max=" + max + "," + super.toString() + ")";
    }

    @Override
    public double getMinDouble() {
        return min;
    }

    @Override
    public double getMaxDouble() {
        return max;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Long;
    }

    public static class Builder extends NumericalColumnAnalysis.Builder<Builder> {
        private long min;
        private long max;

        public Builder min(long min) {
            this.min = min;
            return this;
        }

        public Builder max(long max) {
            this.max = max;
            return this;
        }

        public LongAnalysis build() {
            return new LongAnalysis(this);
        }
    }

}
