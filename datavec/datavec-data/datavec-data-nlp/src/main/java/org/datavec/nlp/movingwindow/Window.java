/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.nlp.movingwindow;

import org.apache.commons.lang3.StringUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


/**
 * A representation of a sliding window.
 * This is used for creating training examples.
 * @author Adam Gibson
 *
 */
public class Window implements Serializable {
    /**
     * 
     */
    private static final long serialVersionUID = 6359906393699230579L;
    private List<String> words;
    private String label = "NONE";
    private boolean beginLabel;
    private boolean endLabel;
    private int median;
    private static String BEGIN_LABEL = "<([A-Z]+|\\d+)>";
    private static String END_LABEL = "</([A-Z]+|\\d+)>";
    private int begin, end;

    /**
     * Creates a window with a context of size 3
     * @param words a collection of strings of size 3
     */
    public Window(Collection<String> words, int begin, int end) {
        this(words, 5, begin, end);

    }

    public String asTokens() {
        return StringUtils.join(words, " ");
    }


    /**
     * Initialize a window with the given size
     * @param words the words to use 
     * @param windowSize the size of the window
     * @param begin the begin index for the window
     * @param end the end index for the window
     */
    public Window(Collection<String> words, int windowSize, int begin, int end) {
        if (words == null)
            throw new IllegalArgumentException("Words must be a list of size 3");

        this.words = new ArrayList<>(words);
        int windowSize1 = windowSize;
        this.begin = begin;
        this.end = end;
        initContext();
    }


    private void initContext() {
        int median = (int) Math.floor(words.size() / 2);
        List<String> begin = words.subList(0, median);
        List<String> after = words.subList(median + 1, words.size());


        for (String s : begin) {
            if (s.matches(BEGIN_LABEL)) {
                this.label = s.replaceAll("(<|>)", "").replace("/", "");
                beginLabel = true;
            } else if (s.matches(END_LABEL)) {
                endLabel = true;
                this.label = s.replaceAll("(<|>|/)", "").replace("/", "");

            }

        }

        for (String s1 : after) {

            if (s1.matches(BEGIN_LABEL)) {
                this.label = s1.replaceAll("(<|>)", "").replace("/", "");
                beginLabel = true;
            }

            if (s1.matches(END_LABEL)) {
                endLabel = true;
                this.label = s1.replaceAll("(<|>)", "");

            }
        }
        this.median = median;

    }



    @Override
    public String toString() {
        return words.toString();
    }

    public List<String> getWords() {
        return words;
    }

    public void setWords(List<String> words) {
        this.words = words;
    }

    public String getWord(int i) {
        return words.get(i);
    }

    public String getFocusWord() {
        return words.get(median);
    }

    public boolean isBeginLabel() {
        return !label.equals("NONE") && beginLabel;
    }

    public boolean isEndLabel() {
        return !label.equals("NONE") && endLabel;
    }

    public String getLabel() {
        return label.replace("/", "");
    }

    public int getWindowSize() {
        return words.size();
    }

    public int getMedian() {
        return median;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public int getBegin() {
        return begin;
    }

    public void setBegin(int begin) {
        this.begin = begin;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }


}
