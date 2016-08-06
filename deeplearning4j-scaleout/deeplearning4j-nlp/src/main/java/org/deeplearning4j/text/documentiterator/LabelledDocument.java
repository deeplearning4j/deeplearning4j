package org.deeplearning4j.text.documentiterator;

import lombok.Data;
import lombok.ToString;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.util.List;

/**
 * This is primitive holder of document, and it's label.
 *
 * @author raver119@gmail.com
 */
@Data
@ToString(exclude="referencedContent")
public class LabelledDocument {
    // initial text representation of current document
    private String content;

    private String label;

    /*
        as soon as sentence was parsed for vocabulary words, there's no need to hold string representation, and we can just stick to references to those VocabularyWords
      */
    private List<VocabWord> referencedContent;
}
