/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.optimize.parameter.integer;

import lombok.NoArgsConstructor;
import org.apache.commons.math3.distribution.IntegerDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.distribution.DistributionUtils;
import org.deeplearning4j.arbiter.optimize.serde.jackson.IntegerDistributionDeserializer;
import org.deeplearning4j.arbiter.optimize.serde.jackson.IntegerDistributionSerializer;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * IntegerParameterSpace is a {@code ParameterSpace<Integer>}; i.e., defines an ordered space of integers between
 * some minimum and maximum value
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"min", "max"})
@NoArgsConstructor
public class IntegerParameterSpace implements ParameterSpace<Integer> {

    @JsonSerialize(using = IntegerDistributionSerializer.class)
    @JsonDeserialize(using = IntegerDistributionDeserializer.class)
    private IntegerDistribution distribution;
    private int index = -1;

    /**
     * Create an IntegerParameterSpace with a uniform distribution between the specified min/max (inclusive)
     *
     * @param min Min value, inclusive
     * @param max Max value, inclusive
     */
    public IntegerParameterSpace(int min, int max) {
        this(new UniformIntegerDistribution(min, max));
    }

    /**
     * Crate an IntegerParametSpace from the given IntegerDistribution
     *
     * @param distribution Distribution to use
     */
    @JsonCreator
    public IntegerParameterSpace(@JsonProperty("distribution") IntegerDistribution distribution) {
        this.distribution = distribution;
    }

    public int getMin() {
        return distribution.getSupportLowerBound();
    }

    public int getMax() {
        return distribution.getSupportUpperBound();
    }

    @Override
    public Integer getValue(double[] input) {
        if (index == -1) {
            throw new IllegalStateException("Cannot get value: ParameterSpace index has not been set");
        }
        return distribution.inverseCumulativeProbability(input[index]);
    }

    @Override
    public int numParameters() {
        return 1;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.singletonList((ParameterSpace) this);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.emptyMap();
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public void setIndices(int... indices) {
        if (indices == null || indices.length != 1)
            throw new IllegalArgumentException("Invalid index");
        this.index = indices[0];
    }

    @Override
    public String toString() {
        if (distribution instanceof UniformIntegerDistribution) {
            return "IntegerParameterSpace(min=" + distribution.getSupportLowerBound() + ",max="
                            + distribution.getSupportUpperBound() + ")";
        } else {
            return "IntegerParameterSpace(" + distribution + ")";
        }
    }

    public boolean equals(Object o) {
        if (o == this)
            return true;
        if (!(o instanceof IntegerParameterSpace))
            return false;
        final IntegerParameterSpace other = (IntegerParameterSpace) o;
        if (!other.canEqual(this))
            return false;
        if (distribution == null ? other.distribution != null
                        : !DistributionUtils.distributionEquals(distribution, other.distribution))
            return false;
        if (this.index != other.index)
            return false;
        return true;
    }

    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + (distribution == null ? 43 : distribution.getClass().hashCode());
        result = result * PRIME + this.index;
        return result;
    }

    protected boolean canEqual(Object other) {
        return other instanceof IntegerParameterSpace;
    }
}
