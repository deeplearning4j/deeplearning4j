package org.ansj.recognition.impl;

import org.ansj.domain.*;
import org.ansj.library.DATDictionary;
import org.ansj.library.DicLibrary;
import org.ansj.recognition.Recognition;
import org.ansj.recognition.arrimpl.ForeignPersonRecognition;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.ansj.util.MathUtil;
import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.tire.domain.SmartForest;
import org.nlpcn.commons.lang.util.WordAlert;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * 词性标注工具类
 * 
 * @author ansj
 * 
 */
public class NatureRecognition implements Recognition {

    private static final long serialVersionUID = 1L;

    private static final Log logger = LogFactory.getLog();

    private static final Forest SUFFIX_FOREST = new Forest();

    private Forest[] forests = null;

    static {
        try (BufferedReader reader = MyStaticValue.getNatureClassSuffix()) {
            String temp = null;
            while ((temp = reader.readLine()) != null) {
                String[] split = temp.split("\t");
                String word = split[0];
                if (word.length() > 1) {
                    word = new StringBuffer(word).reverse().toString();
                }
                SUFFIX_FOREST.add(word, new String[] {split[1]});
            }
        } catch (IOException e) {
            logger.warn("IO异常", e);
        }
    }

    public NatureRecognition() {
        forests = new Forest[] {DicLibrary.get()};
    }

    public NatureRecognition(Forest... forests) {
        this.forests = forests;
    }

    private NatureTerm root = new NatureTerm(TermNature.BEGIN);

    private NatureTerm[] end = {new NatureTerm(TermNature.END)};

    private List<Term> terms = null;

    private NatureTerm[][] natureTermTable = null;

    /**
     * 进行最佳词性查找,引用赋值.所以不需要有返回值
     */
    @Override
    public void recognition(Result result) {
        this.terms = result.getTerms();
        natureTermTable = new NatureTerm[terms.size() + 1][];
        natureTermTable[terms.size()] = end;

        int length = terms.size();
        for (int i = 0; i < length; i++) {
            natureTermTable[i] = getNatureTermArr(terms.get(i).termNatures().termNatures);
        }
        walk();
    }

    /**
     * 传入一组。词对词语进行。词性标注
     * 
     * @param words
     * @param offe
     * @return
     */
    public List<Term> recognition(List<String> words) {
        return recognition(words, 0);
    }

    /**
     * 传入一组。词对词语进行。词性标注
     * 
     * @param words
     * @param offe
     * @return
     */
    public List<Term> recognition(List<String> words, int offe) {
        List<Term> terms = new ArrayList<>(words.size());
        int tempOffe = 0;
        for (String word : words) {
            TermNatures tn = getTermNatures(word);

            terms.add(new Term(word, offe + tempOffe, tn));
            tempOffe += word.length();
        }
        new NatureRecognition().recognition(new Result(terms));
        return terms;
    }

    /**
     * 传入一次词语获得相关的词性
     * 
     * @param word
     * @return
     */
    public TermNatures getTermNatures(String word) {
        String[] params = null;
        // 获得词性 ， 先从系统辞典。在从用户自定义辞典
        AnsjItem ansjItem = DATDictionary.getItem(word);
        TermNatures tn = null;

        if (ansjItem != AnsjItem.NULL) {
            tn = ansjItem.termNatures;
        } else if ((params = getParams(word)) != null) {
            tn = new TermNatures(new TermNature(params[0], 1));
        } else if (WordAlert.isEnglish(word)) {
            tn = TermNatures.EN;
        } else if (WordAlert.isNumber(word)) {
            tn = TermNatures.M;
        } else {
            tn = TermNatures.NULL;
        }
        return tn;
    }

    /**
     * 获取一个词语的参数
     * 
     * @param word
     * @return
     */
    public String[] getParams(String word) {
        for (Forest forest : forests) {
            if (forest == null) {
                continue;
            }
            SmartForest<String[]> sf = forest;
            for (int i = 0; i < word.length(); i++) {
                sf = sf.get(word.charAt(i));
                if (sf == null) {
                    return null;
                }
            }
            if (sf.getStatus() > 1) {
                return sf.getParam();
            } else {
                return null;
            }
        }
        return null;
    }

    /**
     * 通过规则 猜测词性
     * 
     * @param word
     * @return
     */
    public static TermNatures guessNature(String word) {
        String nature = null;
        SmartForest<String[]> smartForest = SUFFIX_FOREST;
        int len = 0;
        for (int i = word.length() - 1; i >= 0; i--) {
            smartForest = smartForest.get(word.charAt(i));
            if (smartForest == null) {
                break;
            }
            len++;
            if (smartForest.getStatus() == 2) {
                nature = smartForest.getParam()[0];
            } else if (smartForest.getStatus() == 3) {
                nature = smartForest.getParam()[0];
                break;
            }
        }

        if ("nt".equals(nature) && (len > 1 || word.length() > 3)) {
            return TermNatures.NT;
        } else if ("ns".equals(nature)) {
            return TermNatures.NS;
        } else if (word.length() < 5) {
            Result parse = ToAnalysis.parse(word);
            for (Term term : parse.getTerms()) {
                if ("nr".equals(term.getNatureStr())) {
                    return TermNatures.NR;
                }
            }
        } else if (ForeignPersonRecognition.isFName(word)) {
            return TermNatures.NRF;
        }

        return TermNatures.NW;
    }

    public void walk() {
        int length = natureTermTable.length - 1;
        setScore(root, natureTermTable[0]);
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < natureTermTable[i].length; j++) {
                setScore(natureTermTable[i][j], natureTermTable[i + 1]);
            }
        }
        optimalRoot();
    }

    private void setScore(NatureTerm natureTerm, NatureTerm[] natureTerms) {

        for (int i = 0; i < natureTerms.length; i++) {
            natureTerms[i].setScore(natureTerm);
        }
    }

    private NatureTerm[] getNatureTermArr(TermNature[] termNatures) {
        NatureTerm[] natureTerms = new NatureTerm[termNatures.length];
        for (int i = 0; i < natureTerms.length; i++) {
            natureTerms[i] = new NatureTerm(termNatures[i]);
        }
        return natureTerms;
    }

    /**
     * 获得最优路径
     */
    private void optimalRoot() {
        NatureTerm to = end[0];
        NatureTerm from = null;
        int index = natureTermTable.length - 1;
        while ((from = to.from) != null && index > 0) {
            terms.get(--index).setNature(from.termNature.nature);
            to = from;
        }
    }

    /**
     * 关于这个term的词性
     * 
     * @author ansj
     * 
     */
    public class NatureTerm {

        public TermNature termNature;

        public double score = 0;

        public double selfScore;

        public NatureTerm from;

        protected NatureTerm(TermNature termNature) {
            this.termNature = termNature;
            selfScore = termNature.frequency + 1;
        }

        public void setScore(NatureTerm natureTerm) {
            double tempScore = MathUtil.compuNatureFreq(natureTerm, this);
            if (from == null || score < tempScore) {
                this.score = tempScore;
                this.from = natureTerm;
            }
        }

        @Override
        public String toString() {
            return termNature.nature.natureStr + "/" + selfScore;
        }

    }
}
