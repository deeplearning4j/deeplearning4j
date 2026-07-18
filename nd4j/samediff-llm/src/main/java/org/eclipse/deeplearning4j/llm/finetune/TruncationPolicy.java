package org.eclipse.deeplearning4j.llm.finetune;

/** Policy used when a rendered example exceeds the configured sequence length. */
public enum TruncationPolicy {
    REJECT,
    RIGHT_TRUNCATE,
    DROP_OLDEST_TURNS
}
