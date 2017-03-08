package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class TokenizerFunction extends BaseTokenizerFunction implements Function<String, Sequence<VocabWord>> {

    public TokenizerFunction(@NonNull Broadcast<VectorsConfiguration> configurationBroadcast) {
        super(configurationBroadcast);
    }


    @Override
    public Sequence<VocabWord> call(String s) throws Exception {
        if (tokenizerFactory == null)
            instantiateTokenizerFactory();

        List<String> tokens = tokenizerFactory.create(s).getTokens();
        Sequence<VocabWord> seq = new Sequence<>();
        for (String token : tokens) {
            if (token == null || token.isEmpty())
                continue;

            seq.addElement(new VocabWord(1.0, token));
        }

        return seq;
    }
}
