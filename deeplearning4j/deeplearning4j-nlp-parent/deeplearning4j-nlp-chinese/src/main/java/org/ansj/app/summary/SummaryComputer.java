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

package org.ansj.app.summary;

import org.ansj.app.keyword.KeyWordComputer;
import org.ansj.app.keyword.Keyword;
import org.ansj.app.summary.pojo.Summary;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.nlpcn.commons.lang.tire.SmartGetWord;
import org.nlpcn.commons.lang.tire.domain.SmartForest;
import org.nlpcn.commons.lang.util.MapCount;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 自动摘要,同时返回关键词
 * 
 * @author ansj
 * 
 */
public class SummaryComputer {

    private static final Set<String> FILTER_SET = new HashSet<>();

    static {
        FILTER_SET.add("w");
        FILTER_SET.add("null");
    }

    /**
     * summaryLength
     */
    private int len = 300;

    private boolean isSplitSummary = true;

    String title, content;

    public SummaryComputer(String title, String content) {
        this.title = title;
        this.content = content;
    }

    public SummaryComputer(int len, String title, String content) {
        this.len = len;
        this.title = title;
        this.content = content;
    }

    public SummaryComputer(int len, boolean isSplitSummary, String title, String content) {
        this.len = len;
        this.title = title;
        this.content = content;
        this.isSplitSummary = isSplitSummary;
    }

    /**
     * 计算摘要，利用关键词抽取计算
     * 
     * @return
     */
    public Summary toSummary() {
        return toSummary(new ArrayList<Keyword>());
    }

    /**
     * 根据用户查询串计算摘要
     * 
     * @return
     */
    public Summary toSummary(String query) {

        List<Term> parse = NlpAnalysis.parse(query).getTerms();

        List<Keyword> keywords = new ArrayList<>();
        for (Term term : parse) {
            if (FILTER_SET.contains(term.natrue().natureStr)) {
                continue;
            }
            keywords.add(new Keyword(term.getName(), term.termNatures().allFreq, 1));
        }

        return toSummary(keywords);
    }

    /**
     * 计算摘要，传入用户自己算好的关键词
     * 
     * @return
     */
    public Summary toSummary(List<Keyword> keywords) {

        if (keywords == null) {
            keywords = new ArrayList<>();
        }

        if (keywords.isEmpty()) {

            KeyWordComputer kc = new KeyWordComputer(10);
            keywords = kc.computeArticleTfidf(title, content);
        }
        return explan(keywords, content);
    }

    /**
     * 计算摘要
     * 
     * @param keyword
     * @param content
     * @return
     */
    private Summary explan(List<Keyword> keywords, String content) {

        SmartForest<Double> sf = new SmartForest<>();

        for (Keyword keyword : keywords) {
            sf.add(keyword.getName(), keyword.getScore());
        }

        // 先断句
        List<Sentence> sentences = toSentenceList(content.toCharArray());

        for (Sentence sentence : sentences) {
            computeScore(sentence, sf);
        }

        double maxScore = 0;
        int maxIndex = 0;

        MapCount<String> mc = new MapCount<>();

        for (int i = 0; i < sentences.size(); i++) {
            double tempScore = sentences.get(i).score;
            int tempLength = sentences.get(i).value.length();
            mc.addAll(sentences.get(i).mc.get());

            if (tempLength >= len) {
                tempScore = tempScore * mc.get().size();
                if (maxScore < tempScore) {
                    maxScore = tempScore;
                    maxIndex = i;
                    continue;
                }
                mc.get().clear();
            }
            for (int j = i + 1; j < sentences.size(); j++) {
                tempScore += sentences.get(j).score;
                tempLength += sentences.get(j).value.length();
                mc.addAll(sentences.get(j).mc.get());

                if (tempLength >= len) {
                    tempScore = tempScore * mc.get().size();
                    if (maxScore < tempScore) {
                        maxScore = tempScore;
                        maxIndex = i;
                    }
                    mc.get().clear();
                    break;
                }
            }

            if (tempLength < len) {
                tempScore = tempScore * mc.get().size();
                if (maxScore < tempScore) {
                    maxScore = tempScore;
                    maxIndex = i;
                    break;
                }
                mc.get().clear();
            }
        }

        StringBuilder sb = new StringBuilder();
        for (int i = maxIndex; i < sentences.size(); i++) {
            sb.append(sentences.get(i).value);
            if (sb.length() > len) {
                break;
            }
        }

        String summaryStr = sb.toString();

        /**
         * 是否强制文本长度。对于abc这种字符算半个长度
         */

        if (isSplitSummary && sb.length() > len) {
            double value = len;

            StringBuilder newSummary = new StringBuilder();
            char c = 0;
            for (int i = 0; i < sb.length(); i++) {
                c = sb.charAt(i);
                if (c < 256) {
                    value -= 0.5;
                } else {
                    value -= 1;
                }

                if (value < 0) {
                    break;
                }

                newSummary.append(c);
            }

            summaryStr = newSummary.toString();
        }

        return new Summary(keywords, summaryStr);
    }

    /**
     * 计算一个句子的分数
     * 
     * @param sentence
     * @param sf
     */
    private void computeScore(Sentence sentence, SmartForest<Double> forest) {
        SmartGetWord<Double> sgw = new SmartGetWord<>(forest, sentence.value);
        String name = null;
        while ((name = sgw.getFrontWords()) != null) {
            sentence.updateScore(name, sgw.getParam());
        }
        if (sentence.score == 0) {
            sentence.score = sentence.value.length() * -0.005;
        } else {
            sentence.score /= Math.log(sentence.value.length() + 3);
        }
    }

    public List<Sentence> toSentenceList(char[] chars) {

        StringBuilder sb = new StringBuilder();

        List<Sentence> sentences = new ArrayList<>();

        for (int i = 0; i < chars.length; i++) {
            if (sb.length() == 0 && (Character.isWhitespace(chars[i]) || chars[i] == ' ')) {
                continue;
            }

            sb.append(chars[i]);
            switch (chars[i]) {
                case '.':
                    if (i < chars.length - 1 && chars[i + 1] > 128) {
                        insertIntoList(sb, sentences);
                        sb = new StringBuilder();
                    }
                    break;
                //case ' ':
                case '	':
                case '　':
                case ' ':
                case ',':
                case '。':
                case ';':
                case '；':
                case '!':
                case '！':
                case '，':
                case '?':
                case '？':
                case '\n':
                case '\r':
                    insertIntoList(sb, sentences);
                    sb = new StringBuilder();
            }
        }

        if (sb.length() > 0) {
            insertIntoList(sb, sentences);
        }

        return sentences;
    }

    private void insertIntoList(StringBuilder sb, List<Sentence> sentences) {
        String content = sb.toString().trim();
        if (content.length() > 0) {
            sentences.add(new Sentence(content));
        }
    }

    /*
     * 句子对象
     */
    public class Sentence {
        String value;
        private double score;

        private MapCount<String> mc = new MapCount<>();

        public Sentence(String value) {
            this.value = value.trim();
        }

        public void updateScore(String name, double score) {
            mc.add(name);
            Double size = mc.get().get(name);
            this.score += score / size;
        }

        @Override
        public String toString() {
            return value;
        }
    }

}
