package org.deeplearning4j.text.tokenization.tokenizer;

import org.deeplearning4j.text.tokenization.tokenizerfactory.JapaneseTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class JapaneseTokenizerTest {

    private String toTokenize = "黒い瞳の綺麗な女の子";
    private String[] expect = {"黒い", "瞳", "の", "綺麗", "な", "女の子"};
    private String baseString = "驚いた彼は道を走っていった。";

    @Test
    public void testJapaneseTokenizer() throws Exception {
        TokenizerFactory t = new JapaneseTokenizerFactory();
        Tokenizer tokenizer = t.create(toTokenize);

        assertEquals(expect.length, tokenizer.countTokens());
        for (int i = 0; i < tokenizer.countTokens(); ++i) {
            assertEquals(tokenizer.nextToken(), expect[i]);
        }
    }

    @Test
    public void testBaseForm() throws Exception {
        TokenizerFactory tf = new JapaneseTokenizerFactory(true);

        Tokenizer tokenizer1 = tf.create(toTokenize);
        Tokenizer tokenizer2 = tf.create(baseString);

        assertEquals("黒い", tokenizer1.nextToken());
        assertEquals("驚く", tokenizer2.nextToken());
    }


    @Test
    public void testGetTokens() throws Exception {
        TokenizerFactory tf = new JapaneseTokenizerFactory();

        Tokenizer tokenizer = tf.create(toTokenize);

        // Exhaust iterator.
        assertEquals(expect.length, tokenizer.countTokens());
        for (int i = 0; i < tokenizer.countTokens(); ++i) {
            assertEquals(tokenizer.nextToken(), expect[i]);
        }

        // Ensure exhausted.
        assertEquals(false, tokenizer.hasMoreTokens());

        // Count doesn't change.
        assertEquals(expect.length, tokenizer.countTokens());

        // getTokens still returns everything.
        List<String> tokens = tokenizer.getTokens();
        assertEquals(expect.length, tokens.size());
    }

    @Test
    public void testKuromojiMultithreading() throws Exception {
        class Worker implements Runnable {
            private final JapaneseTokenizerFactory tf;
            private final String[] jobs;
            private int runs;
            private boolean passed = false;

            public Worker(JapaneseTokenizerFactory tf, String[] jobs, int runs) {
                this.tf = tf;
                this.jobs = jobs;
                this.runs = runs;
            }

            @Override
            public void run() {
                while (runs > 0) {
                    String s = jobs[runs-- % jobs.length];
                    List<String> tokens = tf.create(s).getTokens();
                    StringBuilder sb = new StringBuilder();
                    for (String token : tokens) {
                        sb.append(token);
                    }

                    if (sb.toString().length() != s.length()) {
                        return;
                    }
                }
                passed = true;
            }
        }

        JapaneseTokenizerFactory tf = new JapaneseTokenizerFactory();

        String[] work = {toTokenize, baseString, toTokenize, baseString};
        Worker[] workers = new Worker[10];

        for (int i = 0; i < workers.length; i++) {
            workers[i] = new Worker(tf, work, 50);
        }

        Thread[] threads = new Thread[10];
        for (int i = 0; i < threads.length; i++) {
            threads[i] = new Thread(workers[i]);
            threads[i].start();
        }

        for (Thread thread : threads) {
            thread.join();
        }

        for (int i = 0; i < workers.length; i++) {
            assertTrue(workers[i].passed);
        }
    }
}
