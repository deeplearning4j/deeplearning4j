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

package org.deeplearning4j.arbiter.optimize.parameter.continuous;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.distribution.DistributionUtils;
import org.deeplearning4j.arbiter.optimize.serde.jackson.RealDistributionDeserializer;
import org.deeplearning4j.arbiter.optimize.serde.jackson.RealDistributionSerializer;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * ContinuousParametSpace is a {@code ParameterSpace<Double>} that (optionally) takes an Apache Commons
 * {@link RealDistribution} when used for random sampling (such as in a RandomSearchCandidateGenerator)
 *
 * @author Alex Black
 */
public class ContinuousParameterSpace implements ParameterSpace<Double> {

    //Need to use custom serializers/deserializers for commons RealDistribution instances
    @JsonSerialize(using = RealDistributionSerializer.class)
    @JsonDeserialize(using = RealDistributionDeserializer.class)
    private RealDistribution distribution;
    private int index = -1;

    /**
     * ContinuousParameterSpace with uniform distribution between the minimum and maximum values
     *
     * @param min Minimum value that can be generated
     * @param max Maximum value that can be generated
     */
    public ContinuousParameterSpace(double min, double max) {
        this(new UniformRealDistribution(min, max));
    }

    /**
     * ConditiousParameterSpcae wiht a specified probability distribution. The provided distribution defines the min/max
     * values, and (for random search, etc) will be used when generating random values
     *
     * @param distribution Distribution to sample from
     */
    public ContinuousParameterSpace(@JsonProperty("distribution") RealDistribution distribution) {
        this.distribution = distribution;
    }


    @Override
    public Double getValue(double[] input) {
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
        if (indices == null || indices.length != 1) {
            throw new IllegalArgumentException("Invalid index");
        }
        this.index = indices[0];
    }


    @Override
    public String toString() {
        if (distribution instanceof UniformRealDistribution) {
            return "ContinuousParameterSpace(min=" + distribution.getSupportLowerBound() + ",max="
                            + distribution.getSupportUpperBound() + ")";
        } else {
            return "ContinuousParameterSpace(" + distribution + ")";
        }
    }

    public boolean equals(Object o) {
        if (o == this)
            return true;
        if (!(o instanceof ContinuousParameterSpace))
            return false;
        final ContinuousParameterSpace other = (ContinuousParameterSpace) o;
        if (distribution == null ? other.distribution != null
                        : !DistributionUtils.distributionsEqual(distribution, other.distribution))
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
}
