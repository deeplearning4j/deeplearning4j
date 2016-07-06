package org.nd4j.etl4j.api.transform.analysis.columns;

import io.skymind.echidna.api.ColumnType;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Analysis for Integer columns
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class IntegerAnalysis extends NumericalColumnAnalysis {

    private final int min;
    private final int max;

    private IntegerAnalysis(Builder builder) {
        super(builder);
        this.min = builder.min;
        this.max = builder.max;
    }

    @Override
    public String toString() {
        return "IntegerAnalysis(min=" + min + ",max=" + max + "," + super.toString() + ")";
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
        return ColumnType.Integer;
    }

    public static class Builder extends NumericalColumnAnalysis.Builder<Builder> {
        private int min;
        private int max;

        public Builder min(int min) {
            this.min = min;
            return this;
        }

        public Builder max(int max) {
            this.max = max;
            return this;
        }

        public IntegerAnalysis build() {
            return new IntegerAnalysis(this);
        }
    }
}
