package org.eclipse.deeplearning4j.llm.finetune;

/** Backend-independent token IDs, shifted labels, and assistant-only causal loss mask. */
public final class ResponseMaskedTokens {
    private final int[] inputIds;
    private final int[] labelIds;
    private final float[] lossMask;
    private final int responseStart;
    private final int usedLength;
    private final int originalLength;
    private final boolean truncated;
    private final int droppedMessageCount;
    private final boolean partialAssistant;

    public ResponseMaskedTokens(int[] inputIds, int[] labelIds, float[] lossMask,
                                int responseStart, int usedLength) {
        this(inputIds, labelIds, lossMask, responseStart, usedLength, usedLength, false, 0, false);
    }

    public ResponseMaskedTokens(int[] inputIds, int[] labelIds, float[] lossMask,
                                int responseStart, int usedLength, int originalLength,
                                boolean truncated, int droppedMessageCount, boolean partialAssistant) {
        this.inputIds = inputIds;
        this.labelIds = labelIds;
        this.lossMask = lossMask;
        this.responseStart = responseStart;
        this.usedLength = usedLength;
        this.originalLength = originalLength;
        this.truncated = truncated;
        this.droppedMessageCount = droppedMessageCount;
        this.partialAssistant = partialAssistant;
    }

    public int[] getInputIds() { return inputIds; }
    public int[] getLabelIds() { return labelIds; }
    public float[] getLossMask() { return lossMask; }
    public int getResponseStart() { return responseStart; }
    public int getUsedLength() { return usedLength; }
    /** Token length of the fully rendered conversation before any turn dropping or truncation. */
    public int getOriginalLength() { return originalLength; }
    /** True when any messages were dropped or tokens cut to fit the sequence length. */
    public boolean isTruncated() { return truncated; }
    /** Number of whole conversation messages dropped to fit the sequence length. */
    public int getDroppedMessageCount() { return droppedMessageCount; }
    /** True when the used window ends inside an assistant turn, leaving a partial training target. */
    public boolean isPartialAssistant() { return partialAssistant; }
}
