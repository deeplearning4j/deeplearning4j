package org.eclipse.deeplearning4j.llm.finetune;

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Converts single or multi-turn examples into assistant-masked causal-LM batches. */
public class ResponseMaskedDatasetBuilder {
    private final Tokenizer tokenizer;
    private final int sequenceLength;
    private final TruncationPolicy truncationPolicy;
    private final boolean allowPartialAssistant;

    /** Backward-compatible behavior: right truncate and allow a partial final assistant target. */
    public ResponseMaskedDatasetBuilder(Tokenizer tokenizer, int sequenceLength) {
        this(tokenizer, sequenceLength, TruncationPolicy.RIGHT_TRUNCATE, true);
    }

    public ResponseMaskedDatasetBuilder(Tokenizer tokenizer, int sequenceLength,
                                        TruncationPolicy truncationPolicy,
                                        boolean allowPartialAssistant) {
        if (tokenizer == null) throw new IllegalArgumentException("tokenizer is required");
        if (sequenceLength < 2) throw new IllegalArgumentException("sequenceLength must be at least 2");
        if (truncationPolicy == null) throw new IllegalArgumentException("truncationPolicy is required");
        this.tokenizer = tokenizer;
        this.sequenceLength = sequenceLength;
        this.truncationPolicy = truncationPolicy;
        this.allowPartialAssistant = allowPartialAssistant;
    }

    public List<MultiDataSet> build(List<GeneratedTrainingExample> examples) {
        List<MultiDataSet> result = new ArrayList<>(examples.size());
        for (GeneratedTrainingExample example : examples) result.add(build(example));
        return result;
    }

    public MultiDataSet build(GeneratedTrainingExample example) {
        ResponseMaskedTokens tokens = tokenize(example);
        INDArray inputIds = Nd4j.createFromArray(tokens.getInputIds()).reshape(1, sequenceLength);
        INDArray lossMask = Nd4j.createFromArray(tokens.getLossMask()).reshape(1, sequenceLength);
        INDArray labelIds = Nd4j.createFromArray(tokens.getLabelIds()).reshape(1, sequenceLength);
        return new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{inputIds, lossMask}, new INDArray[]{labelIds});
    }

    /** Prepares all examples without initializing an ND4J backend and retains rejection reasons. */
    public List<TokenPreparationResult> prepare(List<GeneratedTrainingExample> examples) {
        List<TokenPreparationResult> result = new ArrayList<>(examples.size());
        for (GeneratedTrainingExample example : examples) {
            try {
                result.add(TokenPreparationResult.accepted(example.getId(), tokenize(example)));
            } catch (IllegalArgumentException e) {
                result.add(TokenPreparationResult.rejected(
                        example == null ? null : example.getId(), e.getMessage()));
            }
        }
        return result;
    }

    public ResponseMaskedTokens tokenize(GeneratedTrainingExample example) {
        example.validate();
        List<ChatTemplate.Message> messages = toChatMessages(example.effectiveMessages());
        int originalLength = encode(messages, false).length;
        int dropped = 0;
        if (originalLength > sequenceLength && truncationPolicy == TruncationPolicy.REJECT) {
            throw new IllegalArgumentException("Example exceeds sequence length: " + example.getId());
        }
        if (truncationPolicy == TruncationPolicy.DROP_OLDEST_TURNS) {
            while (encode(messages, false).length > sequenceLength && messages.size() > 2) {
                int index = "system".equals(messages.get(0).getRole()) ? 1 : 0;
                messages.remove(index);
                dropped++;
                if (index < messages.size() && "assistant".equals(messages.get(index).getRole())
                        && messages.size() > 1) {
                    messages.remove(index);
                    dropped++;
                }
            }
        }

        int[] full = encode(messages, false);
        int used = Math.min(full.length, sequenceLength);
        boolean[] assistantTokens = new boolean[used];
        int responseStart = used;
        boolean partialAssistant = false;

        for (int messageIndex = 0; messageIndex < messages.size(); messageIndex++) {
            if (!"assistant".equals(messages.get(messageIndex).getRole())) continue;
            List<ChatTemplate.Message> before = new ArrayList<>(messages.subList(0, messageIndex));
            List<ChatTemplate.Message> through = new ArrayList<>(messages.subList(0, messageIndex + 1));
            int start = commonPrefix(encode(before, true), full);
            int rawEnd = messageIndex + 1 == messages.size()
                    ? full.length : commonPrefix(encode(through, false), full);
            if (start < used && rawEnd > used) partialAssistant = true;
            int end = Math.min(rawEnd, used);
            if (start < end) {
                responseStart = Math.min(responseStart, start);
                Arrays.fill(assistantTokens, start, end, true);
            }
        }
        if (responseStart >= used) {
            throw new IllegalArgumentException("All assistant responses were truncated from example " + example.getId());
        }
        if (partialAssistant && !allowPartialAssistant) {
            throw new IllegalArgumentException("Assistant response was partially truncated from example " + example.getId());
        }

        int pad = tokenizer.getPadTokenId() >= 0 ? tokenizer.getPadTokenId() : 0;
        int[] input = new int[sequenceLength];
        int[] labels = new int[sequenceLength];
        float[] mask = new float[sequenceLength];
        Arrays.fill(input, pad);
        Arrays.fill(labels, pad);
        System.arraycopy(full, 0, input, 0, used);
        for (int i = 0; i + 1 < used; i++) {
            labels[i] = full[i + 1];
            if (assistantTokens[i + 1]) mask[i] = 1.0f;
        }
        boolean truncated = dropped > 0 || full.length > sequenceLength;
        return new ResponseMaskedTokens(input, labels, mask, responseStart, used,
                originalLength, truncated, dropped, partialAssistant);
    }

    private int[] encode(List<ChatTemplate.Message> messages, boolean generationPrompt) {
        return tokenizer.encode(tokenizer.applyChatTemplate(messages, generationPrompt), false).getIds();
    }

    private static List<ChatTemplate.Message> toChatMessages(List<FineTuneMessage> source) {
        List<ChatTemplate.Message> result = new ArrayList<>(source.size());
        for (FineTuneMessage message : source) {
            result.add(new ChatTemplate.Message(message.getRole(), message.getContent()));
        }
        return result;
    }

    private static int commonPrefix(int[] a, int[] b) {
        int length = Math.min(a.length, b.length);
        int i = 0;
        while (i < length && a[i] == b[i]) i++;
        return i;
    }
}
