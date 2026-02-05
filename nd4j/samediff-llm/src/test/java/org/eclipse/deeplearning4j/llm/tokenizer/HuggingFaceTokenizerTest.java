package org.eclipse.deeplearning4j.llm.tokenizer;

import org.junit.Test;
import static org.junit.Assert.*;

public class HuggingFaceTokenizerTest {

    @Test
    public void testNativeAvailable() {
        System.out.println("Native available: " + HuggingFaceTokenizer.isNativeAvailable());
        System.out.println("Native version: " + HuggingFaceTokenizer.getNativeVersion());
        assertTrue("Native tokenizers should be available", HuggingFaceTokenizer.isNativeAvailable());
    }
}
