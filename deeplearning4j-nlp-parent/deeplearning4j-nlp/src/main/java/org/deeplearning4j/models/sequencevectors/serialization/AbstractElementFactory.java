package org.deeplearning4j.models.sequencevectors.serialization;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceElementFactory;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * This is universal serialization/deserialization factor for SequenceVectors serialization.
 * It will work for any &lt;T extends SequenceElement&gt; that doesn't breaks simple POJO rules.
 *
 * @author raver119@gmail.com
 */
public class AbstractElementFactory<T extends SequenceElement> implements SequenceElementFactory<T> {
    private final Class targetClass;

    protected static final Logger log = LoggerFactory.getLogger(AbstractElementFactory.class);

    /**
     * This is the only constructor available for AbstractElementFactory
     * @param cls class that going to be serialized/deserialized using this instance. I.e.: VocabWord.class
     */
    public AbstractElementFactory(@NonNull Class<? extends SequenceElement> cls) {
        targetClass = cls;
    }

    /**
     * This method builds object from provided JSON
     *
     * @param json JSON for restored object
     * @return restored object
     */
    @Override
    public T deserialize(String json) {
        ObjectMapper mapper = SequenceElement.mapper();
        try {
            T ret = (T) mapper.readValue(json, targetClass);
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
    public String serialize(T element) {
        String json = null;
        try {
            json = element.toJSON();
        } catch (Exception e) {
            log.error("Direct serialization failed, falling back to jackson");
        }

        if (json == null || json.isEmpty()) {
            ObjectMapper mapper = SequenceElement.mapper();
            try {
                json = mapper.writeValueAsString(element);
            } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        }

        return json;
    }
}
