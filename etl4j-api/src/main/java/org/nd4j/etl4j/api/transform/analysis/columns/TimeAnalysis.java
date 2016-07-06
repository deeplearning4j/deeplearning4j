package org.nd4j.etl4j.api.transform.analysis.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;
import io.skymind.echidna.api.ColumnType;
import org.joda.time.DateTimeZone;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

/**
 * Analysis for Time columns
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class TimeAnalysis extends NumericalColumnAnalysis {

    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("YYYY-MM-dd HH:mm:ss.SSS zzz").withZone(DateTimeZone.UTC);
    private final long min;
    private final long max;

    private TimeAnalysis(Builder builder) {
        super(builder);
        this.min = builder.min;
        this.max = builder.max;
    }

    @Override
    public String toString() {
        return "TimeAnalysis(min=" + min + " (" + formatter.print(min) + "),max=" + max + " (" + formatter.print(max) + ")," + super.toString() + ")";
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

        public TimeAnalysis build() {
            return new TimeAnalysis(this);
        }
    }

}
