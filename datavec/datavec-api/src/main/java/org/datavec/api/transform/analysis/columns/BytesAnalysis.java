/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.analysis.columns;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;

/**
 * Analysis for bytes (byte[]) columns
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
@NoArgsConstructor //For Jackson deserialization
public class BytesAnalysis implements ColumnAnalysis {

    private long countTotal;
    private long countNull;
    private long countZeroLength;
    private int minNumBytes;
    private int maxNumBytes;

    public BytesAnalysis(Builder builder) {
        this.countTotal = builder.countTotal;
        this.countNull = builder.countNull;
        this.countZeroLength = builder.countZeroLength;
        this.minNumBytes = builder.minNumBytes;
        this.maxNumBytes = builder.maxNumBytes;
    }


    @Override
    public String toString() {
        return "BytesAnalysis()";
    }


    @Override
    public long getCountTotal() {
        return countTotal;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Bytes;
    }

    public static class Builder {
        private long countTotal;
        private long countNull;
        private long countZeroLength;
        private int minNumBytes;
        private int maxNumBytes;

        public Builder countTotal(long countTotal) {
            this.countTotal = countTotal;
            return this;
        }

        public Builder countNull(long countNull) {
            this.countNull = countNull;
            return this;
        }

        public Builder countZeroLength(long countZeroLength) {
            this.countZeroLength = countZeroLength;
            return this;
        }

        public Builder minNumBytes(int minNumBytes) {
            this.minNumBytes = minNumBytes;
            return this;
        }

        public Builder maxNumBytes(int maxNumBytes) {
            this.maxNumBytes = maxNumBytes;
            return this;
        }

        public BytesAnalysis build() {
            return new BytesAnalysis(this);
        }
    }

}
