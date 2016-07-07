package org.datavec.api.transform.dataquality.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Quality of an Integer column
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class IntegerQuality extends ColumnQuality {

    private final long countNonInteger;

    public IntegerQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countNonInteger){
        super(countValid,countInvalid,countMissing,countTotal);
        this.countNonInteger = countNonInteger;
    }


    public IntegerQuality add(IntegerQuality other){
        return new IntegerQuality(
                countValid + other.countValid,
                countInvalid + other.countInvalid,
                countMissing + other.countMissing,
                countTotal + other.countTotal,
                countNonInteger + other.countNonInteger);
    }

    @Override
    public String toString(){
        return "IntegerQuality(" + super.toString() + ", countNonInteger=" + countNonInteger + ")";
    }

}
