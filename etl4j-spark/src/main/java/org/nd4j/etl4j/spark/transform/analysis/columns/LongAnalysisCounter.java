package org.nd4j.etl4j.spark.transform.analysis.columns;

import org.nd4j.etl4j.spark.transform.analysis.AnalysisCounter;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.etl4j.api.writable.Writable;

import java.math.BigInteger;

/**
 * A counter function for doing analysis on Long columns, on Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor @Data
public class LongAnalysisCounter implements AnalysisCounter<LongAnalysisCounter> {

    private long countZero;
    private long countPositive;
    private long countNegative;
    private long countMinValue;
    private long minValueSeen = Long.MAX_VALUE;
    private long countMaxValue;
    private long maxValueSeen = Long.MIN_VALUE;
    private BigInteger sum = BigInteger.ZERO;
    private long countTotal = 0;

    public LongAnalysisCounter(){

    }

    @Override
    public LongAnalysisCounter add(Writable writable) {
        long value = writable.toLong();

        if(value == 0) countZero++;
        else if(value < 0) countNegative++;
        else countPositive++;

        if(value == minValueSeen){
            countMinValue++;
        } else if( value < minValueSeen ){
            //New minimum value
            minValueSeen = value;
            countMinValue = 1;
        } //Don't need an else condition: if value > minValueSeen, no change to min value or count

        if(value == maxValueSeen){
            countMaxValue++;
        } else if(value > maxValueSeen){
            //new maximum value
            maxValueSeen = value;
            countMaxValue = 1;
        } //Don't need else condition: if value < maxValueSeen, no change to max value or count

        sum = sum.add(BigInteger.valueOf(value));
        countTotal++;

        return this;
    }

    public LongAnalysisCounter merge(LongAnalysisCounter other){
        if(minValueSeen == other.minValueSeen){
            countMinValue += other.countMinValue;
        } else if(minValueSeen > other.minValueSeen) {
            //Keep other, take count from other
            minValueSeen = other.minValueSeen;
            countMinValue = other.countMinValue;
        } //else: Keep this min, no change to count

        if(maxValueSeen == other.maxValueSeen){
            countMaxValue += other.countMaxValue;
        } else if(maxValueSeen < other.maxValueSeen) {
            //Keep other, take count from other
            maxValueSeen = other.maxValueSeen;
            countMaxValue = other.countMaxValue;
        } //else: Keep this max, no change to count

        countZero += other.countZero;
        countPositive += other.countPositive;
        countNegative += other.countNegative;
        sum = sum.add(other.sum);
        countTotal += other.countTotal;

        return this;
    }

}
