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

package org.ansj.app.keyword;

import org.ansj.domain.Term;
import org.ansj.splitWord.Analysis;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.nlpcn.commons.lang.util.StringUtil;

import java.util.*;

public class KeyWordComputer<T extends Analysis> {

    private static final Map<String, Double> POS_SCORE = new HashMap<>();
    private T analysisType;


    static {
        POS_SCORE.put("null", 0.0);
        POS_SCORE.put("w", 0.0);
        POS_SCORE.put("en", 0.0);
        POS_SCORE.put("m", 0.0);
        POS_SCORE.put("num", 0.0);
        POS_SCORE.put("nr", 3.0);
        POS_SCORE.put("nrf", 3.0);
        POS_SCORE.put("nw", 3.0);
        POS_SCORE.put("nt", 3.0);
        POS_SCORE.put("l", 0.2);
        POS_SCORE.put("a", 0.2);
        POS_SCORE.put("nz", 3.0);
        POS_SCORE.put("v", 0.2);
        POS_SCORE.put("kw", 6.0); //关键词词性
    }

    private int nKeyword = 5;


    public KeyWordComputer() {}

    public void setAnalysisType(T analysisType) {
        this.analysisType = analysisType;
    }


    /**
     * 返回关键词个数
     *
     * @param nKeyword
     */
    public KeyWordComputer(int nKeyword) {
        this.nKeyword = nKeyword;
        this.analysisType = (T) new NlpAnalysis();//默认使用NLP的分词方式

    }

    public KeyWordComputer(int nKeyword, T analysisType) {
        this.nKeyword = nKeyword;
        this.analysisType = analysisType;
    }

    /**
     * @param content 正文
     * @return
     */
    private List<Keyword> computeArticleTfidf(String content, int titleLength) {
        Map<String, Keyword> tm = new HashMap<>();

        List<Term> parse = analysisType.parseStr(content).getTerms();
        //FIXME: 这个依赖于用户自定义词典的词性,所以得需要另一个方法..
        //		parse = FilterModifWord.updateNature(parse) ;

        for (Term term : parse) {
            double weight = getWeight(term, content.length(), titleLength);
            if (weight == 0)
                continue;

            Keyword keyword = tm.get(term.getName());


            if (keyword == null) {
                keyword = new Keyword(term.getName(), term.natrue().allFrequency, weight);
                tm.put(term.getName(), keyword);
            } else {
                keyword.updateWeight(1);
            }
        }

        TreeSet<Keyword> treeSet = new TreeSet<>(tm.values());

        ArrayList<Keyword> arrayList = new ArrayList<>(treeSet);
        if (treeSet.size() <= nKeyword) {
            return arrayList;
        } else {
            return arrayList.subList(0, nKeyword);
        }

    }

    /**
     * @param title   标题
     * @param content 正文
     * @return
     */
    public List<Keyword> computeArticleTfidf(String title, String content) {
        if (StringUtil.isBlank(title)) {
            title = "";
        }
        if (StringUtil.isBlank(content)) {
            content = "";
        }
        return computeArticleTfidf(title + "\t" + content, title.length());
    }

    /**
     * 只有正文
     *
     * @param content
     * @return
     */
    public List<Keyword> computeArticleTfidf(String content) {
        return computeArticleTfidf(content, 0);
    }

    private double getWeight(Term term, int length, int titleLength) {
        if (term.getName().trim().length() < 2) {
            return 0;
        }

        String pos = term.natrue().natureStr;

        Double posScore = POS_SCORE.get(pos);

        if (posScore == null) {
            posScore = 1.0;
        } else if (posScore == 0) {
            return 0;
        }

        if (titleLength > term.getOffe()) {
            return 5 * posScore;
        }
        return (length - term.getOffe()) * posScore / length;
    }

}
