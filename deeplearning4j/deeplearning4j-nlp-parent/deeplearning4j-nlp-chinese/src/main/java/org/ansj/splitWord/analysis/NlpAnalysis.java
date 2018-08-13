package org.ansj.splitWord.analysis;

import org.ansj.app.crf.SplitWord;
import org.ansj.dic.LearnTool;
import org.ansj.domain.*;
import org.ansj.library.CrfLibrary;
import org.ansj.recognition.arrimpl.*;
import org.ansj.recognition.impl.NatureRecognition;
import org.ansj.splitWord.Analysis;
import org.ansj.util.AnsjReader;
import org.ansj.util.Graph;
import org.ansj.util.NameFix;
import org.ansj.util.TermUtil;
import org.ansj.util.TermUtil.InsertTermType;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.util.MapCount;
import org.nlpcn.commons.lang.util.WordAlert;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.Reader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 自然语言分词,具有未登录词发现功能。建议在自然语言理解中用。搜索中不要用
 * 
 * @author ansj
 * 
 */
public class NlpAnalysis extends Analysis {

    private static final Log LOG = LogFactory.getLog(NlpAnalysis.class);

    private LearnTool learn = null;

    private static final String TAB = "\t";

    private static final int CRF_WEIGHT = 6;

    private SplitWord splitWord = CrfLibrary.get();

    @Override
    protected List<Term> getResult(final Graph graph) {

        Merger merger = new Merger() {
            @Override
            public List<Term> merger() {

                if (learn == null) {
                    learn = new LearnTool();
                }

                graph.walkPath();

                learn.learn(graph, splitWord, forests);

                // 姓名识别
                if (graph.hasPerson && isNameRecognition) {
                    // 亚洲人名识别
                    new AsianPersonRecognition().recognition(graph.terms);
                    graph.walkPathByScore();
                    NameFix.nameAmbiguity(graph.terms);
                    // 外国人名识别
                    new ForeignPersonRecognition().recognition(graph.terms);
                    graph.walkPathByScore();
                }

                if (splitWord != null) {
                    MapCount<String> mc = new MapCount<>();

                    // 通过crf分词
                    List<String> words = splitWord.cut(graph.chars);

                    Term tempTerm = null;

                    int tempOff = 0;

                    if (!words.isEmpty()) {
                        String word = words.get(0);
                        if (!isRuleWord(word)) {
                            mc.add("始##始" + TAB + word, CRF_WEIGHT);
                        }
                    }

                    for (String word : words) {

                        TermNatures termNatures = new NatureRecognition(forests).getTermNatures(word); // 尝试从词典获取词性

                        Term term = null;

                        if (termNatures != TermNatures.NULL) {
                            term = new Term(word, tempOff, termNatures);
                        } else {
                            term = new Term(word, tempOff, TermNatures.NW);
                            term.setNewWord(true);
                        }

                        tempOff += word.length(); // 增加偏移量
                        if (isRuleWord(word)) { // 如果word不对那么不要了
                            tempTerm = null;
                            continue;
                        }

                        if (term.isNewWord()) { // 尝试猜测词性
                            termNatures = NatureRecognition.guessNature(word);
                            term.updateTermNaturesAndNature(termNatures);
                        }

                        TermUtil.insertTerm(graph.terms, term, InsertTermType.SCORE_ADD_SORT);

                        // 对于非词典中的词持有保守态度
                        if (tempTerm != null && !tempTerm.isNewWord() && !term.isNewWord()) {
                            mc.add(tempTerm.getName() + TAB + word, CRF_WEIGHT);
                        }

                        tempTerm = term;

                        if (term.isNewWord()) {
                            learn.addTerm(new NewWord(word, Nature.NW));
                        }

                    }

                    if (tempTerm != null && !tempTerm.isNewWord()) {
                        mc.add(tempTerm.getName() + TAB + "末##末", CRF_WEIGHT);
                    }
                    graph.walkPath(mc.get());
                } else {
                    LOG.warn("not find any crf model, make sure your config right? ");
                }

                // 数字发现
                if (graph.hasNum && isNumRecognition) {
                    new NumRecognition().recognition(graph.terms);
                }

                // 词性标注
                List<Term> result = getResult();

                // 用户自定义词典的识别
                new UserDefineRecognition(InsertTermType.SCORE_ADD_SORT, forests).recognition(graph.terms);
                graph.rmLittlePath();
                graph.walkPathByScore();

                // 进行新词发现
                new NewWordRecognition(learn).recognition(graph.terms);
                graph.walkPathByScore();

                // 优化后重新获得最优路径
                result = getResult();

                // 激活辞典
                for (Term term : result) {
                    learn.active(term.getName());
                }

                setRealName(graph, result);

                return result;
            }

            private List<Term> getResult() {

                List<Term> result = new ArrayList<>();
                int length = graph.terms.length - 1;
                for (int i = 0; i < length; i++) {
                    if (graph.terms[i] == null) {
                        continue;
                    }
                    result.add(graph.terms[i]);
                }
                return result;
            }
        };
        return merger.merger();
    }

    // 临时处理新词中的特殊字符
    private static final Set<Character> filter = new HashSet<>();

    static {
        filter.add(':');
        filter.add(' ');
        filter.add('：');
        filter.add('　');
        filter.add('，');
        filter.add('”');
        filter.add('“');
        filter.add('？');
        filter.add('。');
        filter.add('！');
        filter.add('。');
        filter.add(',');
        filter.add('.');
        filter.add('、');
        filter.add('\\');
        filter.add('；');
        filter.add(';');
        filter.add('？');
        filter.add('?');
        filter.add('!');
        filter.add('\"');
        filter.add('（');
        filter.add('）');
        filter.add('(');
        filter.add(')');
        filter.add('…');
        filter.add('…');
        filter.add('—');
        filter.add('-');
        filter.add('－');

        filter.add('—');
        filter.add('《');
        filter.add('》');

    }

    /**
     * 判断新词识别出来的词是否可信
     * 
     * @param word
     * @return
     */
    public static boolean isRuleWord(String word) {
        char c = 0;
        for (int i = 0; i < word.length(); i++) {
            c = word.charAt(i);

            if (c != '·') {
                if (c < 256 || filter.contains(c) || (c = WordAlert.CharCover(word.charAt(i))) > 0) {
                    return true;
                }
            }
        }
        return false;
    }

    public NlpAnalysis setCrfModel(SplitWord splitWord) {
        this.splitWord = splitWord;
        return this;
    }

    public NlpAnalysis setLearnTool(LearnTool learn) {
        this.learn = learn;
        return this;
    }

    public NlpAnalysis() {
        super();
    }

    public NlpAnalysis(Reader reader) {
        super.resetContent(new AnsjReader(reader));
    }

    public static Result parse(String str) {
        return new NlpAnalysis().parseStr(str);
    }

    public static Result parse(String str, Forest... forests) {
        return new NlpAnalysis().setForests(forests).parseStr(str);
    }

}
