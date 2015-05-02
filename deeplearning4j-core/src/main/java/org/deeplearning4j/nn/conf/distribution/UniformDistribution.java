package org.deeplearning4j.nn.conf.distribution;

import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class UniformDistribution extends Distribution {

    private static final long serialVersionUID = 7006579116682205405L;

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
            throw new NumberIsTooLargeException(
                    LocalizedFormats.LOWER_BOUND_NOT_BELOW_UPPER_BOUND,
                    lower, upper, false);
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
        if (Double.doubleToLongBits(lower) != Double
                .doubleToLongBits(other.lower))
            return false;
        if (Double.doubleToLongBits(upper) != Double
                .doubleToLongBits(other.upper))
            return false;
        return true;
    }

    public String toString() {
        return "UniformDistribution{" +
                "lower=" + lower +
                ", upper=" + upper +
                '}';
    }
}
