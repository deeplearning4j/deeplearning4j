package org.datavec.api.transform.dataquality.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Created by Alex on 5/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class StringQuality extends ColumnQuality {

    private final long countEmptyString;    //"" string
    private final long countAlphabetic; //A-Z, a-z only
    private final long countNumerical;  //0-9 only
    private final long countWordCharacter;   //A-Z, a-z, 0-9
    private final long countWhitespace;  //tab, spaces etc ONLY
    private final long countUnique;

    public StringQuality(){
        this(0,0,0,0,0,0,0,0,0,0);
    }

    public StringQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countEmptyString,
                         long countAlphabetic, long countNumerical, long countWordCharacter, long countWhitespace,
                              long countUnique){
        super(countValid,countInvalid,countMissing,countTotal);
        this.countEmptyString = countEmptyString;
        this.countAlphabetic = countAlphabetic;
        this.countNumerical = countNumerical;
        this.countWordCharacter = countWordCharacter;
        this.countWhitespace = countWhitespace;
        this.countUnique = countUnique;
    }


    public StringQuality add(StringQuality other){
        return new StringQuality(
                countValid + other.countValid,
                countInvalid + other.countInvalid,
                countMissing + other.countMissing,
                countTotal + other.countTotal,
                countEmptyString + other.countEmptyString,
                countAlphabetic + other.countAlphabetic,
                countNumerical + other.countNumerical,
                countWordCharacter + other.countWordCharacter,
                countWhitespace + other.countWhitespace,
                countUnique + other.countUnique);
    }

    @Override
    public String toString(){
        return "StringQuality(" + super.toString() + ", countEmptyString=" + countEmptyString
                + ", countAlphabetic=" + countAlphabetic + ", countNumerical=" + countNumerical +
                ", countWordCharacter=" + countWordCharacter + ", countWhitespace=" + countWhitespace
                + ", countUnique=" + countUnique + ")";
    }

}
