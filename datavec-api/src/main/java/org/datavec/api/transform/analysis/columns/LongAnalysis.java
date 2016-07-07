package org.datavec.api.transform.analysis.columns;

import org.datavec.api.transform.ColumnType;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Analysis for Long columns
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class LongAnalysis extends NumericalColumnAnalysis {

    private final long min;
    private final long max;

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
