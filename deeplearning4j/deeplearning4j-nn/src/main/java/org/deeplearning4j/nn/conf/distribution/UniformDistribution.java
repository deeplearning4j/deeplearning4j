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

package org.deeplearning4j.nn.conf.distribution;

import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A uniform distribution.
 *
 */
public class UniformDistribution extends Distribution {

    private double upper, lower;

    /**
     * Create a uniform real distribution using the given lower and upper
     * bounds.
     *
     * @param lower Lower bound of this distribution (inclusive).
     * @param upper Upper bound of this distribution (exclusive).
     * @throws NumberIsTooLargeException if {@code lower >= upper}.
     */
    @JsonCreator
    public UniformDistribution(@JsonProperty("lower") double lower, @JsonProperty("upper") double upper)
                    throws NumberIsTooLargeException {
        if (lower >= upper) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND, lower, upper,
                            false);
        }
        this.lower = lower;
        this.upper = upper;
    }

    public double getUpper() {
        return upper;
    }

    public void setUpper(double upper) {
        this.upper = upper;
    }

    public double getLower() {
        return lower;
    }

    public void setLower(double lower) {
        this.lower = lower;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        long temp;
        temp = Double.doubleToLongBits(lower);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(upper);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        UniformDistribution other = (UniformDistribution) obj;
        if (Double.doubleToLongBits(lower) != Double.doubleToLongBits(other.lower))
            return false;
        if (Double.doubleToLongBits(upper) != Double.doubleToLongBits(other.upper))
            return false;
        return true;
    }

    public String toString() {
        return "UniformDistribution{" + "lower=" + lower + ", upper=" + upper + '}';
    }
}
