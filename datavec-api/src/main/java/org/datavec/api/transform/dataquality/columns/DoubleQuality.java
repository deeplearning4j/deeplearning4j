package org.datavec.api.transform.dataquality.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Created by Alex on 5/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class DoubleQuality extends ColumnQuality {

    private final long countNonReal;
    private final long countNaN;
    private final long countInfinite;

    public DoubleQuality(){
        this(0,0,0,0,0,0,0);
    }

    public DoubleQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countNonReal,
                         long countNaN, long countInfinite){
        super(countValid,countInvalid,countMissing,countTotal);
        this.countNonReal = countNonReal;
        this.countNaN = countNaN;
        this.countInfinite = countInfinite;
    }


    public DoubleQuality add(DoubleQuality other){
        return new DoubleQuality(
                countValid + other.countValid,
                countInvalid + other.countInvalid,
                countMissing + other.countMissing,
                countTotal + other.countTotal,
                countNonReal + other.countNonReal,
                countNaN + other.countNaN,
                countInfinite + other.countInfinite);
    }

    @Override
    public String toString(){
        return "DoubleQuality(" + super.toString() + ", countNonReal=" + countNonReal + ", countNaN="
                + countNaN + ", countInfinite=" + countInfinite + ")";
    }

}
