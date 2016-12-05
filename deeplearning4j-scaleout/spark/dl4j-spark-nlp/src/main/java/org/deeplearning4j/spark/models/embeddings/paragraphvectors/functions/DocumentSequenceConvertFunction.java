package org.deeplearning4j.spark.models.embeddings.paragraphvectors.functions;

import lombok.NonNull;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.spark.models.embeddings.sequencevectors.functions.BaseTokenizerFunction;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import scala.Tuple2;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class DocumentSequenceConvertFunction extends BaseTokenizerFunction implements Function<LabelledDocument, Sequence<VocabWord>> {

    public DocumentSequenceConvertFunction(@NonNull Broadcast<VectorsConfiguration> configurationBroadcast) {
        super(configurationBroadcast);
    }

    @Override
    public Sequence<VocabWord> call(LabelledDocument document) throws Exception {
        Sequence<VocabWord> sequence = new Sequence<>();

        // get elements
        if (document.getReferencedContent() != null && !document.getReferencedContent().isEmpty()) {
            sequence.addElements(document.getReferencedContent());
        } else {
            if (tokenizerFactory == null)
                instantiateTokenizerFactory();

            List<String> tokens = tokenizerFactory.create(document.getContent()).getTokens();

            for (String token: tokens) {
                if (token == null || token.isEmpty())
                    continue;

                VocabWord word = new VocabWord(1.0, token);
                sequence.addElement(word);
            }
        }

        // get labels
        for (String label: document.getLabels()) {
            if (label == null || label.isEmpty())
                continue;

            VocabWord labelElement = new VocabWord(1.0, label);
            labelElement.markAsLabel(true);

            sequence.addSequenceLabel(labelElement);
        }

        return sequence;
    }
}
