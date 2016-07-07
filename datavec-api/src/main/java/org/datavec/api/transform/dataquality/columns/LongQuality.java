package org.datavec.api.transform.dataquality.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Quality of a Long column
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class LongQuality extends ColumnQuality {

    private final long countNonLong;

    public LongQuality(){
        this(0,0,0,0,0);
    }

    public LongQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countNonLong){
        super(countValid,countInvalid,countMissing,countTotal);
        this.countNonLong = countNonLong;
    }


    public LongQuality add(LongQuality other){
        return new LongQuality(
                countValid + other.countValid,
                countInvalid + other.countInvalid,
                countMissing + other.countMissing,
                countTotal + other.countTotal,
                countNonLong + other.countNonLong);
    }

    @Override
    public String toString(){
        return "LongQuality(" + super.toString() + ", countNonLong=" + countNonLong + ")";
    }

}
