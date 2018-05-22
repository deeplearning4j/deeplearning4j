package org.deeplearning4j.models.sequencevectors.serialization;

import org.deeplearning4j.models.sequencevectors.interfaces.SequenceElementFactory;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;

/**
 * @author raver119@gmail.com
 */
public class VocabWordFactory implements SequenceElementFactory<VocabWord> {

    /**
     * This method builds object from provided JSON
     *
     * @param json JSON for restored object
     * @return restored object
     */
    @Override
    public VocabWord deserialize(String json) {
        ObjectMapper mapper = SequenceElement.mapper();
        try {
            VocabWord ret = mapper.readValue(json, VocabWord.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method serializaes object  into JSON string
     *
     * @param element
     * @return
     */
    @Override
    public String serialize(VocabWord element) {
        return element.toJSON();
    }
}
