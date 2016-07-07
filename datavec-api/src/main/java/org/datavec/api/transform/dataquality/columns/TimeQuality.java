package org.datavec.api.transform.dataquality.columns;

/**
 * TimeQuality: quality of a time column
 *
 * @author Alex Black
 */
public class TimeQuality extends ColumnQuality {


    public TimeQuality(long countValid, long countInvalid, long countMissing, long countTotal) {
        super(countValid, countInvalid, countMissing, countTotal);
    }

    public TimeQuality(){
        this(0,0,0,0);
    }

    @Override
    public String toString(){
        return "TimeQuality(" + super.toString() + ")";
    }

    public TimeQuality add(TimeQuality other){
        return new TimeQuality(
                countValid + other.countValid,
                countInvalid + other.countInvalid,
                countMissing + other.countMissing,
                countTotal + other.countTotal);
    }
}
