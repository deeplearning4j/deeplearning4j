package org.nd4j.etl4j.api.transform.transform.time;

import org.nd4j.etl4j.api.transform.MathOp;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.metadata.TimeMetaData;
import org.nd4j.etl4j.api.transform.transform.BaseColumnTransform;
import org.nd4j.etl4j.api.io.data.LongWritable;
import org.nd4j.etl4j.api.writable.Writable;

import java.util.concurrent.TimeUnit;

/**
 * Transform math op on a time column
 *
 * Note: only the following MathOps are supported: Add, Subtract, ScalarMin, ScalarMax<br>
 * For ScalarMin/Max, the TimeUnit must be milliseconds - i.e., value must be in epoch millisecond format
 *
 * @author Alex Black
 */
public class TimeMathOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final long timeQuantity;
    private final TimeUnit timeUnit;
    private final long asMilliseconds;

    public TimeMathOpTransform(String columnName, MathOp mathOp, long timeQuantity, TimeUnit timeUnit){
        super(columnName);
        if(mathOp != MathOp.Add && mathOp != MathOp.Subtract && mathOp != MathOp.ScalarMin && mathOp != MathOp.ScalarMax){
            throw new IllegalArgumentException("Invalid MathOp: only Add/Subtract/ScalarMin/ScalarMax supported");
        }
        if((mathOp == MathOp.ScalarMin || mathOp == MathOp.ScalarMax) && timeUnit != TimeUnit.MILLISECONDS){
            throw new IllegalArgumentException("Only valid time unit for ScalarMin/Max is Milliseconds (i.e., timestamp format)");
        }

        this.mathOp = mathOp;
        this.timeQuantity = timeQuantity;
        this.timeUnit = timeUnit;
        this.asMilliseconds = TimeUnit.MILLISECONDS.convert(timeQuantity,timeUnit);
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnType) {
        if(!(oldColumnType instanceof TimeMetaData)) throw new IllegalStateException("Cannot execute TimeMathOpTransform on column with type " + oldColumnType);
        return new TimeMetaData(((TimeMetaData) oldColumnType).getTimeZone());
    }

    @Override
    public Writable map(Writable columnWritable) {
        long currTime = columnWritable.toLong();
        switch (mathOp){
            case Add:
                return new LongWritable(currTime + asMilliseconds);
            case Subtract:
                return new LongWritable(currTime - asMilliseconds);
            case ScalarMax:
                return new LongWritable(Math.max(asMilliseconds,currTime));
            case ScalarMin:
                return new LongWritable(Math.min(asMilliseconds,currTime));
            default:
                throw new RuntimeException("Invalid MathOp for TimeMathOpTransform: " + mathOp);
        }
    }

    @Override
    public String toString() {
        return "TimeMathOpTransform(mathOp=" + mathOp + "," + timeQuantity + "-" + timeUnit + ")";
    }
}
