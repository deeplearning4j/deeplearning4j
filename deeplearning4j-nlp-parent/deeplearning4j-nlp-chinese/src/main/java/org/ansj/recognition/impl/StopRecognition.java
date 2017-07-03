package org.ansj.recognition.impl;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.recognition.Recognition;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

/**
 * 对结果增加过滤,支持词性过滤,和词语过滤.
 * 
 * @author Ansj
 *
 */
public class StopRecognition implements Recognition {

    private static final Log LOG = LogFactory.getLog();

    /**
     * 
     */
    private static final long serialVersionUID = 7041503137429986566L;

    private Set<String> stop = new HashSet<String>();

    private Set<String> natureStop = new HashSet<String>();

    private Set<Pattern> regexList = new HashSet<Pattern>();

    /**
     * 批量增加停用词
     * 
     * @param filterWords
     * @return
     */
    public StopRecognition insertStopWords(Collection<String> filterWords) {
        stop.addAll(filterWords);
        return this;
    }

    /**
     * 批量增加停用词
     * 
     * @param stopWords
     * @return
     */
    public StopRecognition insertStopWords(String... stopWords) {
        for (String words : stopWords) {
            stop.add(words);
        }
        return this;
    }

    /**
     * 批量增加停用词性 比如 增加nr 后.人名将不在结果中
     * 
     * @param stopWords
     */
    public void insertStopNatures(String... stopNatures) {
        for (String natureStr : stopNatures) {
            natureStop.add(natureStr);
        }
    }

    /**
     * 增加正则表达式过滤
     * 
     * @param regex
     */
    public void insertStopRegexes(String... regexes) {
        for (String regex : regexes) {
            try {
                regexList.add(Pattern.compile(regex));
            } catch (Exception e) {
                e.printStackTrace();
                LOG.error("regex err : " + regex, e);
            }
        }

    }

    @Override
    public void recognition(Result result) {
        List<Term> list = result.getTerms();
        Iterator<Term> iterator = list.iterator();

        while (iterator.hasNext()) {
            Term term = iterator.next();
            if (filter(term)) {
                iterator.remove();
            }
        }

    }

    /**
     * 判断一个词语是否停用..
     * 
     * @param term
     * @return
     */
    public boolean filter(Term term) {

        if (stop.size() > 0 && (stop.contains(term.getName()))) {
            return true;
        }

        if (natureStop.size() > 0 && (natureStop.contains(term.natrue().natureStr))) {
            return true;
        }

        if (regexList.size() > 0) {
            for (Pattern stopwordPattern : regexList) {
                if (stopwordPattern.matcher(term.getName()).matches()) {
                    return true;
                }
            }
        }

        return false;
    }

    public void clear() {
        this.stop.clear();
        this.natureStop.clear();
        this.regexList.clear();
    }

}
