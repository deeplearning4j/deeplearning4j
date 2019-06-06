package org.ansj.recognition.arrimpl;

import org.ansj.domain.*;
import org.ansj.library.NgramLibrary;
import org.ansj.recognition.TermArrRecognition;
import org.ansj.util.TermUtil;
import org.ansj.util.TermUtil.InsertTermType;

import java.util.ArrayList;
import java.util.List;

/**
 * 人名识别工具类
 * 
 * @author ansj
 * 
 */
public class AsianPersonRecognition implements TermArrRecognition {
    private static final double[] FACTORY = {0.16271366224044456, 0.8060521860870434, 0.031234151672511947};
    private boolean skip = false;
    private Term[] terms;

    // 名称是否有歧异
    // public int B = -1;//0 姓氏
    // public int C = -1;//1 双名的首字
    // public int D = -1;//2 双名的末字
    // public int E = -1;//3 单名
    // public int N = -1; //4任意字
    // public int L = -1;//11 人名的下文
    // public int M = -1;//12 两个中国人名之间的成分
    // public int m = -1;//44 可拆分的姓名
    // double[] factory = {"BC", "BCD", "BCDE"}

    @Override
    public void recognition(Term[] terms) {
        this.terms = terms;
        List<Term> termList = recogntion_();
        for (Term term2 : termList) {
            TermUtil.insertTerm(terms, term2, InsertTermType.SCORE_ADD_SORT);
        }
    }

    private List<Term> recogntion_() {
        Term term = null;
        Term tempTerm = null;
        List<Term> termList = new ArrayList<>();
        int beginFreq = 10;
        for (int i = 0; i < terms.length; i++) {
            term = terms[i];
            if (term == null || !term.termNatures().personAttr.flag) {
                continue;
            }
            term.score(0);
            term.selfScore(0);
            int freq = 0;
            for (int j = 2; j > -1; j--) {
                freq = term.termNatures().personAttr.getFreq(j, 0);
                if ((freq > 10) || (term.getName().length() == 2 && freq > 10)) {
                    tempTerm = nameFind(i, beginFreq, j);
                    if (tempTerm != null) {
                        termList.add(tempTerm);
                        // 如果是无争议性识别
                        if (skip) {
                            for (int j2 = i; j2 < tempTerm.toValue(); j2++) {
                                if (terms[j2] != null) {
                                    terms[j2].score(0);
                                    terms[j2].selfScore(0);
                                }
                            }
                            i = tempTerm.toValue() - 1;
                            break;
                        }
                    }
                }
            }
            beginFreq = term.termNatures().personAttr.begin + 1;
        }
        return termList;
    }

    /**
     * 人名识别
     * 
     * @param term
     * @param offe
     * @param freq
     */

    private Term nameFind(int offe, int beginFreq, int size) {
        StringBuilder sb = new StringBuilder();
        int undefinite = 0;
        skip = false;
        PersonNatureAttr pna = null;
        int index = 0;
        int freq = 0;
        double allFreq = 0;
        Term term = null;
        int i = offe;
        for (; i < terms.length; i++) {
            // 走到结尾处识别出来一个名字.
            if (terms[i] == null) {
                continue;
            }
            term = terms[i];
            pna = term.termNatures().personAttr;
            // 在这个长度的这个位置的词频,如果没有可能就干掉,跳出循环
            if ((freq = pna.getFreq(size, index)) == 0) {
                return null;
            }

            if (pna.allFreq > 0) {
                undefinite++;
            }
            sb.append(term.getName());
            allFreq += Math.log(term.termNatures().allFreq + 1);
            allFreq += -Math.log((freq));
            index++;

            if (index == size + 2) {
                break;
            }
        }

        double score = -Math.log(FACTORY[size]);
        score += allFreq;
        double endFreq = 0;
        // 开始寻找结尾词
        boolean flag = true;
        while (flag) {
            i++;
            if (i >= terms.length) {
                endFreq = 10;
                flag = false;
            } else if (terms[i] != null) {
                int twoWordFreq = NgramLibrary.getTwoWordFreq(term, terms[i]);
                if (twoWordFreq > 3) {
                    return null;
                }
                endFreq = terms[i].termNatures().personAttr.end + 1;
                flag = false;
            }
        }

        score -= Math.log(endFreq);
        score -= Math.log(beginFreq);

        if (score > -3) {
            return null;
        }

        if (allFreq > 0 && undefinite > 0) {
            return null;
        }

        skip = undefinite == 0;
        term = new Term(sb.toString(), offe, TermNatures.NR);
        term.selfScore(score);

        return term;

    }

    public List<NewWord> getNewWords(Term[] terms) {
        this.terms = terms;
        List<NewWord> all = new ArrayList<>();
        List<Term> termList = recogntion_();
        for (Term term2 : termList) {
            all.add(new NewWord(term2.getName(), Nature.NR));
        }
        return all;
    }

    public List<Term> getNewTerms() {
        return recogntion_();
    }

}
