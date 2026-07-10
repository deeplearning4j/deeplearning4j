package org.eclipse.deeplearning4j.llm.finetune;

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Converts generated examples into causal-LM batches with exact assistant-response masks. */
public class ResponseMaskedDatasetBuilder {
    private final Tokenizer tokenizer;
    private final int sequenceLength;

    public ResponseMaskedDatasetBuilder(Tokenizer tokenizer, int sequenceLength) {
        if (tokenizer == null) throw new IllegalArgumentException("tokenizer is required");
        if (sequenceLength < 2) throw new IllegalArgumentException("sequenceLength must be at least 2");
        this.tokenizer = tokenizer;
        this.sequenceLength = sequenceLength;
    }

    public List<MultiDataSet> build(List<GeneratedTrainingExample> examples) {
        List<MultiDataSet> result = new ArrayList<>(examples.size());
        for (GeneratedTrainingExample example : examples) result.add(build(example));
        return result;
    }

    public MultiDataSet build(GeneratedTrainingExample example) {
        example.validate();
        List<ChatTemplate.Message> prefixMessages = new ArrayList<>();
        if (example.getSystemPrompt() != null && !example.getSystemPrompt().trim().isEmpty()) {
            prefixMessages.add(ChatTemplate.Message.system(example.getSystemPrompt()));
        }
        prefixMessages.add(ChatTemplate.Message.user(example.getPrompt()));
        int[] prefix = tokenizer.encode(tokenizer.applyChatTemplate(prefixMessages, true), false).getIds();

        List<ChatTemplate.Message> fullMessages = new ArrayList<>(prefixMessages);
        fullMessages.add(ChatTemplate.Message.assistant(example.getResponse()));
        int[] full = tokenizer.encode(tokenizer.applyChatTemplate(fullMessages, false), false).getIds();
        int responseStart = commonPrefix(prefix, full);
        int used = Math.min(full.length, sequenceLength);
        if (responseStart >= used) {
            throw new IllegalArgumentException("Response was truncated from example " + example.getId());
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
            if (i + 1 >= responseStart) mask[i] = 1.0f;
        }

        INDArray inputIds = Nd4j.createFromArray(input).reshape(1, sequenceLength);
        INDArray lossMask = Nd4j.createFromArray(mask).reshape(1, sequenceLength);
        INDArray labelIds = Nd4j.createFromArray(labels).reshape(1, sequenceLength);
        return new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{inputIds, lossMask}, new INDArray[]{labelIds});
    }

    private static int commonPrefix(int[] a, int[] b) {
        int length = Math.min(a.length, b.length);
        int i = 0;
        while (i < length && a[i] == b[i]) i++;
        return i;
    }
}
