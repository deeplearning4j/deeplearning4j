package org.deeplearning4j.models.sequencevectors.serialization;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY, getterVisibility = JsonAutoDetect.Visibility.NONE,
        setterVisibility = JsonAutoDetect.Visibility.NONE)
@Data
public class ExtVocabWord extends VocabWord {
    protected String extString;
    protected long counter;

    protected ExtVocabWord() {
        super();
    }

    public ExtVocabWord(String extString, long counter, double wordFrequency, @NonNull String word) {
        super(wordFrequency, word);
        this.extString = extString;
        this.counter = counter;
    }
}
