package org.deeplearning4j.spark.models.embeddings.sequencevectors.functions;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class TokenizerFunction implements Function<String, Sequence<VocabWord>> {
    protected Broadcast<VectorsConfiguration> configurationBroadcast;

    protected transient TokenizerFactory tokenizerFactory;
    protected transient TokenPreProcess tokenPreprocessor;

    public TokenizerFunction(@NonNull Broadcast<VectorsConfiguration> configurationBroadcast) {
        this.configurationBroadcast = configurationBroadcast;
    }

    public void instantiateTokenizerFactory() {
        String tfClassName = this.configurationBroadcast.getValue().getTokenizerFactory();
        String tpClassName = this.configurationBroadcast.getValue().getTokenPreProcessor();

        if (tfClassName != null && !tfClassName.isEmpty()) {
            try {
                tokenizerFactory = (TokenizerFactory) Class.forName(tfClassName).newInstance();

                if (tpClassName != null && !tpClassName.isEmpty()) {
                    try {
                        tokenPreprocessor = (TokenPreProcess) Class.forName(tpClassName).newInstance();
                    } catch (Exception e) {
                        throw new RuntimeException("Unable to instantiate TokenPreProcessor.", e);
                    }
                }

                if (tokenPreprocessor != null) {
                    tokenizerFactory.setTokenPreProcessor(tokenPreprocessor);
                }
            } catch (Exception e ) {
                throw new RuntimeException("Unable to instantiate TokenizerFactory.", e);
            }
        } else throw new RuntimeException("TokenizerFactory wasn't defined.");
    }

    @Override
    public Sequence<VocabWord> call(String s) throws Exception {
        if (tokenizerFactory == null)
            instantiateTokenizerFactory();

        List<String> tokens =  tokenizerFactory.create(s).getTokens();
        Sequence<VocabWord> seq = new Sequence<>();
        for (String token: tokens) {
            if (token == null)
                continue;

            seq.addElement(new VocabWord(1.0, token));
        }

        return seq;
    }
}
