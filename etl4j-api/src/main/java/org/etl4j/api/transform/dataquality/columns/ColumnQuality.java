package io.skymind.echidna.api.dataquality.columns;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/**
 * Base class for the quality of a column
 *
 * @author Alex Black
 */
@AllArgsConstructor @Data
public abstract class ColumnQuality implements Serializable {

    protected final long countValid;
    protected final long countInvalid;
    protected final long countMissing;
    protected final long countTotal;


    @Override
    public String toString(){
        return "countValid=" + countValid + ", countInvalid=" + countInvalid + ", countMissing=" + countMissing
                + ", countTotal=" + countTotal;
    }
}
