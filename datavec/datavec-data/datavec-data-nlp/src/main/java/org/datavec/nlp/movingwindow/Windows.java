/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.nlp.movingwindow;


import org.apache.commons.lang3.StringUtils;
import org.datavec.nlp.tokenization.tokenizer.DefaultStreamTokenizer;
import org.datavec.nlp.tokenization.tokenizer.Tokenizer;
import org.datavec.nlp.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Static utility class for textual based windowing functions
 * @author Adam Gibson
 */
public class Windows {


    /**
     * Constructs a list of window of size windowSize.
     * Note that padding for each window is created as well.
     * @param words the words to tokenize and construct windows from
     * @param windowSize the window size to generate
     * @return the list of windows for the tokenized string
     */
    public static List<Window> windows(InputStream words, int windowSize) {
        Tokenizer tokenizer = new DefaultStreamTokenizer(words);
        List<String> list = new ArrayList<>();
        while (tokenizer.hasMoreTokens())
            list.add(tokenizer.nextToken());
        return windows(list, windowSize);
    }

    /**
     * Constructs a list of window of size windowSize.
     * Note that padding for each window is created as well.
     * @param words the words to tokenize and construct windows from
     * @param tokenizerFactory tokenizer factory to use
     * @param windowSize the window size to generate
     * @return the list of windows for the tokenized string
     */
    public static List<Window> windows(InputStream words, TokenizerFactory tokenizerFactory, int windowSize) {
        Tokenizer tokenizer = tokenizerFactory.create(words);
        List<String> list = new ArrayList<>();
        while (tokenizer.hasMoreTokens())
            list.add(tokenizer.nextToken());

        if (list.isEmpty())
            throw new IllegalStateException("No tokens found for windows");

        return windows(list, windowSize);
    }


    /**
     * Constructs a list of window of size windowSize.
     * Note that padding for each window is created as well.
     * @param words the words to tokenize and construct windows from
     * @param windowSize the window size to generate
     * @return the list of windows for the tokenized string
     */
    public static List<Window> windows(String words, int windowSize) {
        StringTokenizer tokenizer = new StringTokenizer(words);
        List<String> list = new ArrayList<String>();
        while (tokenizer.hasMoreTokens())
            list.add(tokenizer.nextToken());
        return windows(list, windowSize);
    }

    /**
     * Constructs a list of window of size windowSize.
     * Note that padding for each window is created as well.
     * @param words the words to tokenize and construct windows from
     * @param tokenizerFactory tokenizer factory to use
     * @param windowSize the window size to generate
     * @return the list of windows for the tokenized string
     */
    public static List<Window> windows(String words, TokenizerFactory tokenizerFactory, int windowSize) {
        Tokenizer tokenizer = tokenizerFactory.create(words);
        List<String> list = new ArrayList<>();
        while (tokenizer.hasMoreTokens())
            list.add(tokenizer.nextToken());

        if (list.isEmpty())
            throw new IllegalStateException("No tokens found for windows");

        return windows(list, windowSize);
    }


    /**
     * Constructs a list of window of size windowSize.
     * Note that padding for each window is created as well.
     * @param words the words to tokenize and construct windows from
     * @return the list of windows for the tokenized string
     */
    public static List<Window> windows(String words) {
        StringTokenizer tokenizer = new StringTokenizer(words);
        List<String> list = new ArrayList<String>();
        while (tokenizer.hasMoreTokens())
            list.add(tokenizer.nextToken());
        return windows(list, 5);
    }

    /**
     * Constructs a list of window of size windowSize.
     * Note that padding for each window is created as well.
     * @param words the words to tokenize and construct windows from
     * @param tokenizerFactory tokenizer factory to use
     * @return the list of windows for the tokenized string
     */
    public static List<Window> windows(String words, TokenizerFactory tokenizerFactory) {
        Tokenizer tokenizer = tokenizerFactory.create(words);
        List<String> list = new ArrayList<>();
        while (tokenizer.hasMoreTokens())
            list.add(tokenizer.nextToken());
        return windows(list, 5);
    }


    /**
     * Creates a sliding window from text
     * @param windowSize the window size to use
     * @param wordPos the position of the word to center
     * @param sentence the sentence to createComplex a window for
     * @return a window based on the given sentence
     */
    public static Window windowForWordInPosition(int windowSize, int wordPos, List<String> sentence) {
        List<String> window = new ArrayList<>();
        List<String> onlyTokens = new ArrayList<>();
        int contextSize = (int) Math.floor((windowSize - 1) / 2);

        for (int i = wordPos - contextSize; i <= wordPos + contextSize; i++) {
            if (i < 0)
                window.add("<s>");
            else if (i >= sentence.size())
                window.add("</s>");
            else {
                onlyTokens.add(sentence.get(i));
                window.add(sentence.get(i));

            }
        }

        String wholeSentence = StringUtils.join(sentence);
        String window2 = StringUtils.join(onlyTokens);
        int begin = wholeSentence.indexOf(window2);
        int end = begin + window2.length();
        return new Window(window, begin, end);

    }


    /**
     * Constructs a list of window of size windowSize
     * @param words the words to  construct windows from
     * @return the list of windows for the tokenized string
     */
    public static List<Window> windows(List<String> words, int windowSize) {

        List<Window> ret = new ArrayList<>();

        for (int i = 0; i < words.size(); i++)
            ret.add(windowForWordInPosition(windowSize, i, words));


        return ret;
    }

}
