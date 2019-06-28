package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import lombok.NonNull;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.base.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * CompositePreProcessor is a {@link TokenPreProcess} that applies multiple preprocessors sequentially
 * @author Alex Black
 */
public class CompositePreProcessor implements TokenPreProcess {

    private List<TokenPreProcess> preProcessors;

    public CompositePreProcessor(@NonNull TokenPreProcess... preProcessors){
        Preconditions.checkState(preProcessors.length > 0, "No preprocessors were specified (empty input)");
        this.preProcessors = Arrays.asList(preProcessors);
    }

    public CompositePreProcessor(@NonNull Collection<? extends TokenPreProcess> preProcessors){
        Preconditions.checkState(!preProcessors.isEmpty(), "No preprocessors were specified (empty input)");
        this.preProcessors = new ArrayList<>(preProcessors);
    }

    @Override
    public String preProcess(String token) {
        String s = token;
        for(TokenPreProcess tpp : preProcessors){
            s = tpp.preProcess(s);
        }
        return s;
    }
}
