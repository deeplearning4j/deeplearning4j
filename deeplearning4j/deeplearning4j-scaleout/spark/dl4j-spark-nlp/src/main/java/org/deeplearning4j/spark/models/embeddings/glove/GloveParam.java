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

package org.deeplearning4j.spark.models.embeddings.glove;

import org.apache.spark.broadcast.Broadcast;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.primitives.CounterMap;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class GloveParam implements Serializable {

    private int vectorLength;
    private boolean useAdaGrad;
    private double lr;
    private Random gen;
    private double negative;
    private double xMax;
    private double maxCount;
    private Broadcast<CounterMap<String, String>> coOccurrenceCounts;

    public GloveParam(int vectorLength, boolean useAdaGrad, double lr, Random gen, double negative, double xMax,
                    double maxCount, Broadcast<CounterMap<String, String>> coOccurrenceCounts) {
        this.vectorLength = vectorLength;
        this.useAdaGrad = useAdaGrad;
        this.lr = lr;
        this.gen = gen;
        this.negative = negative;
        this.xMax = xMax;
        this.maxCount = maxCount;
        this.coOccurrenceCounts = coOccurrenceCounts;
    }

    public int getVectorLength() {
        return vectorLength;
    }

    public void setVectorLength(int vectorLength) {
        this.vectorLength = vectorLength;
    }

    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }

    public double getLr() {
        return lr;
    }

    public void setLr(double lr) {
        this.lr = lr;
    }

    public Random getGen() {
        return gen;
    }

    public void setGen(Random gen) {
        this.gen = gen;
    }

    public double getNegative() {
        return negative;
    }

    public void setNegative(double negative) {
        this.negative = negative;
    }

    public double getxMax() {
        return xMax;
    }

    public void setxMax(double xMax) {
        this.xMax = xMax;
    }

    public double getMaxCount() {
        return maxCount;
    }

    public void setMaxCount(double maxCount) {
        this.maxCount = maxCount;
    }

    public Broadcast<CounterMap<String, String>> getCoOccurrenceCounts() {
        return coOccurrenceCounts;
    }

    public void setCoOccurrenceCounts(Broadcast<CounterMap<String, String>> coOccurrenceCounts) {
        this.coOccurrenceCounts = coOccurrenceCounts;
    }


    public static class Builder {
        private int vectorLength = 300;
        private boolean useAdaGrad = true;
        private double lr = 0.025;
        private Random gen;
        private double negative = 5;
        private double xMax = 0.75;
        private double maxCount = 100;
        private Broadcast<CounterMap<String, String>> coOccurrenceCounts;

        public Builder vectorLength(int vectorLength) {
            this.vectorLength = vectorLength;
            return this;
        }

        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder lr(double lr) {
            this.lr = lr;
            return this;
        }

        public Builder gen(Random gen) {
            this.gen = gen;
            return this;
        }

        public Builder negative(double negative) {
            this.negative = negative;
            return this;
        }

        public Builder xMax(double xMax) {
            this.xMax = xMax;
            return this;
        }

        public Builder maxCount(double maxCount) {
            this.maxCount = maxCount;
            return this;
        }

        public Builder coOccurrenceCounts(Broadcast<CounterMap<String, String>> coOccurrenceCounts) {
            this.coOccurrenceCounts = coOccurrenceCounts;
            return this;
        }

        public GloveParam build() {
            return new GloveParam(vectorLength, useAdaGrad, lr, gen, negative, xMax, maxCount, coOccurrenceCounts);
        }
    }

}
