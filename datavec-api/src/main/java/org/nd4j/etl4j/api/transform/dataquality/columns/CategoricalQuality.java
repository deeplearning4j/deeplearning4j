package org.nd4j.etl4j.api.transform.dataquality.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Quality of a Categorical column
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class CategoricalQuality extends ColumnQuality {

    public CategoricalQuality(){
        super(0,0,0,0);
    }

    public CategoricalQuality(long countValid, long countInvalid, long countMissing, long countTotal){
        super(countValid,countInvalid,countMissing,countTotal);
    }

    public CategoricalQuality add(CategoricalQuality other){
        return new CategoricalQuality(
                countValid + other.countValid,
                countInvalid + other.countInvalid,
                countMissing + other.countMissing,
                countTotal + other.countTotal);
    }

    @Override
    public String toString(){
        return "CategoricalQuality(" + super.toString() + ")";
    }

}
