package io.skymind.echidna.spark.quality.string;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.canova.api.io.data.NullWritable;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.dataquality.columns.StringQuality;
import io.skymind.echidna.api.metadata.StringMetaData;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class StringQualityAddFunction implements Function2<StringQuality,Writable,StringQuality> {

    private final StringMetaData meta;

    @Override
    public StringQuality call(StringQuality v1, Writable writable) throws Exception {
        long valid = v1.getCountValid();
        long invalid = v1.getCountInvalid();
        long countMissing = v1.getCountMissing();
        long countTotal = v1.getCountTotal() + 1;
        long empty = v1.getCountEmptyString();
        long alphabetic = v1.getCountAlphabetic();
        long numerical = v1.getCountNumerical();
        long word = v1.getCountWordCharacter();
        long whitespaceOnly = v1.getCountWhitespace();

        String str = writable.toString();

        if(writable instanceof NullWritable) countMissing++;
        else if(meta.isValid(writable)) valid++;
        else invalid++;

        if(str == null || str.isEmpty()){
            empty++;
        } else {
            if(str.matches("[a-zA-Z]")) alphabetic++;
            if(str.matches("\\d+")) numerical++;
            if(str.matches("\\w+")) word++;
            if(str.matches("\\s+")) whitespaceOnly++;
        }

        return new StringQuality(valid,invalid,countMissing,countTotal,empty,alphabetic,numerical,word,whitespaceOnly,0);
    }
}
