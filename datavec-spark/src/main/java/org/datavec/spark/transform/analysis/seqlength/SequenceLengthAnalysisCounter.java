package org.datavec.spark.transform.analysis.seqlength;

import org.datavec.spark.transform.analysis.AnalysisCounter;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.writable.Writable;

/**
 * Created by Alex on 7/03/2016.
 */
@AllArgsConstructor @Data
public class SequenceLengthAnalysisCounter implements AnalysisCounter<SequenceLengthAnalysisCounter> {

    private long countZeroLength;
    private long countOneLength;
    private long countMinLength;
    private int minLengthSeen = Integer.MAX_VALUE;
    private long countMaxLength;
    private int maxLengthSeen = Integer.MIN_VALUE;
    private long countTotal;
    private double mean;


    public SequenceLengthAnalysisCounter(){

    }

    @Override
    public SequenceLengthAnalysisCounter add(Writable writable) {
        return this;
    }

    public SequenceLengthAnalysisCounter merge(SequenceLengthAnalysisCounter other){
        int otherMin = other.getMinLengthSeen();
        int newMinLengthSeen;
        long newCountMinValue;
        if(minLengthSeen == otherMin){
            newMinLengthSeen = minLengthSeen;
            newCountMinValue = countMinLength + other.countMinLength;
        } else if(minLengthSeen > otherMin) {
            //Keep other, take count from other
            newMinLengthSeen = otherMin;
            newCountMinValue = other.countMinLength;
        } else {
            //Keep this min, no change to count
            newMinLengthSeen = minLengthSeen;
            newCountMinValue = countMinLength;
        }

        int otherMax = other.getMaxLengthSeen();
        int newMaxLengthSeen;
        long newCountMaxValue;
        if(maxLengthSeen == otherMax){
            newMaxLengthSeen = maxLengthSeen;
            newCountMaxValue = countMaxLength + other.countMaxLength;
        } else if(maxLengthSeen < otherMax) {
            //Keep other, take count from other
            newMaxLengthSeen = otherMax;
            newCountMaxValue = other.countMaxLength;
        } else {
            //Keep this max, no change to count
            newMaxLengthSeen = maxLengthSeen;
            newCountMaxValue = countMaxLength;
        }

        //Calculate the new mean, in an online fashion:
        long newCountTotal = countTotal + other.countTotal;
        double sum = countTotal*mean + other.countTotal*other.mean;
        double newMean = sum / newCountTotal;


        return new SequenceLengthAnalysisCounter(countZeroLength+other.countZeroLength,
                countOneLength + other.countOneLength,
                newCountMinValue,
                newMinLengthSeen,
                newCountMaxValue,
                newMaxLengthSeen,
                newCountTotal,
                newMean);
    }

}
