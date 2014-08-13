package org.deeplearning4j.linalg.util;

import java.io.Serializable;

/**
 * Iteration result for iterating through a dimension
 *
 * @author Adan Gibson
 */
public class IterationResult implements Serializable {
    private double result;
    private boolean nextSlice;


    public IterationResult(double result, boolean nextSlice) {
        this.result = result;
        this.nextSlice = nextSlice;
    }

    public double getResult() {
        return result;
    }

    public void setResult(double result) {
        this.result = result;
    }

    public boolean isNextSlice() {
        return nextSlice;
    }

    public void setNextSlice(boolean nextSlice) {
        this.nextSlice = nextSlice;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof IterationResult)) return false;

        IterationResult that = (IterationResult) o;

        if (nextSlice != that.nextSlice) return false;
        if (Double.compare(that.result, result) != 0) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result1;
        long temp;
        temp = Double.doubleToLongBits(result);
        result1 = (int) (temp ^ (temp >>> 32));
        result1 = 31 * result1 + (nextSlice ? 1 : 0);
        return result1;
    }
}
