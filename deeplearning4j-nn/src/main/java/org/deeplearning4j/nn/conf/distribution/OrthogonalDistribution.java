/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.nn.conf.distribution;

import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Orthogonal distribution.
 *
 */
public class OrthogonalDistribution extends Distribution {

    private double gain;

    /**
     * Create a log-normal distribution
     * with the given mean and std
     *
     * @param gain the gain
     */
    @JsonCreator
    public OrthogonalDistribution(@JsonProperty("gain") double gain) {
        this.gain = gain;
    }

    public double getGain() {
        return gain;
    }

    public void setGain(double gain) {
        this.gain = gain;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        long temp;
        temp = Double.doubleToLongBits(gain);
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
        OrthogonalDistribution other = (OrthogonalDistribution) obj;
        if (Double.doubleToLongBits(gain) != Double.doubleToLongBits(other.gain))
            return false;
        return true;
    }

    public String toString() {
        return "OrthogonalDistribution{gain=" + gain + "}";
    }
}
