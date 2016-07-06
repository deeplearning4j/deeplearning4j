package io.skymind.echidna.spark.analysis.columns;

import io.skymind.echidna.spark.analysis.AnalysisCounter;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.canova.api.writable.Writable;

/**
 * A counter function for doing analysis on integer columns, on Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor @Data
public class IntegerAnalysisCounter implements AnalysisCounter<IntegerAnalysisCounter> {

    private long countZero;
    private long countPositive;
    private long countNegative;
    private long countMinValue;
    private int minValueSeen = Integer.MAX_VALUE;
    private long countMaxValue;
    private int maxValueSeen = Integer.MIN_VALUE;
    private long sum = 0;
    private long countTotal = 0;



    public IntegerAnalysisCounter(){

    }


    @Override
    public IntegerAnalysisCounter add(Writable writable) {
        int value = writable.toInt();

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

        sum += value;
        countTotal++;

        return this;
    }

    public IntegerAnalysisCounter merge(IntegerAnalysisCounter other){
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
        sum += other.sum;
        countTotal += other.countTotal;

        return this;
    }

}
