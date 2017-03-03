package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseTokenizerFunction implements Serializable {
    protected Broadcast<VectorsConfiguration> configurationBroadcast;

    protected transient TokenizerFactory tokenizerFactory;
    protected transient TokenPreProcess tokenPreprocessor;

    protected BaseTokenizerFunction(@NonNull Broadcast<VectorsConfiguration> configurationBroadcast) {
        this.configurationBroadcast = configurationBroadcast;
    }

    protected void instantiateTokenizerFactory() {
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
            } catch (Exception e) {
                throw new RuntimeException("Unable to instantiate TokenizerFactory.", e);
            }
        } else
            throw new RuntimeException("TokenizerFactory wasn't defined.");
    }
}
