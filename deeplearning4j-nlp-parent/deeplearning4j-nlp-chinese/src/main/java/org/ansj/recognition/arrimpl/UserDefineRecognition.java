package org.ansj.recognition.arrimpl;

import org.ansj.domain.Term;
import org.ansj.domain.TermNature;
import org.ansj.domain.TermNatures;
import org.ansj.library.DicLibrary;
import org.ansj.recognition.TermArrRecognition;
import org.ansj.util.TermUtil;
import org.ansj.util.TermUtil.InsertTermType;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.tire.domain.SmartForest;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

/**
 * 用户自定义词典.又称补充词典
 * 
 * @author ansj
 * 
 */
public class UserDefineRecognition implements TermArrRecognition {

    public static final Log logger = LogFactory.getLog(UserDefineRecognition.class);

    private Term[] terms = null;

    private Forest[] forests = {DicLibrary.get()};

    private int offe = -1;
    private int endOffe = -1;
    private int tempFreq = 50;
    private String tempNature;

    private SmartForest<String[]> branch = null;
    private SmartForest<String[]> forest = null;

    private InsertTermType type = InsertTermType.SKIP;

    public UserDefineRecognition(InsertTermType type, Forest... forests) {
        this.type = type;
        if (forests != null && forests.length > 0) {
            this.forests = forests;
        }

    }

    @Override
    public void recognition(Term[] terms) {
        this.terms = terms;
        for (Forest forest : forests) {
            if (forest == null) {
                continue;
            }
            reset();
            this.forest = forest;

            branch = forest;

            int length = terms.length - 1;

            boolean flag = true;
            for (int i = 0; i < length; i++) {
                if (terms[i] == null)
                    continue;
                if (branch == forest) {
                    flag = false;
                } else {
                    flag = true;
                }

                branch = termStatus(branch, terms[i]);
                if (branch == null) {
                    if (offe != -1) {
                        i = offe;
                    }
                    reset();
                } else if (branch.getStatus() == 3) {
                    endOffe = i;
                    tempNature = branch.getParam()[0];
                    tempFreq = getInt(branch.getParam()[1], 50);
                    if (offe != -1 && offe < endOffe) {
                        i = offe;
                        makeNewTerm();
                        reset();
                    } else {
                        reset();
                    }
                } else if (branch.getStatus() == 2) {
                    endOffe = i;
                    if (offe == -1) {
                        offe = i;
                    } else {
                        tempNature = branch.getParam()[0];
                        tempFreq = getInt(branch.getParam()[1], 50);
                        if (flag) {
                            makeNewTerm();
                        }
                    }
                } else if (branch.getStatus() == 1) {
                    if (offe == -1) {
                        offe = i;
                    }
                }
            }
            if (offe != -1 && offe < endOffe) {
                makeNewTerm();
            }
        }
    }

    private int getInt(String str, int def) {
        try {
            return Integer.parseInt(str);
        } catch (NumberFormatException e) {
            logger.warn(str + "不是一个数字", e);
            return def;
        }
    }

    private void makeNewTerm() {
        StringBuilder sb = new StringBuilder();
        for (int j = offe; j <= endOffe; j++) {
            if (terms[j] == null) {
                continue;
            } else {
                sb.append(terms[j].getName());
            }
        }
        TermNatures termNatures = new TermNatures(new TermNature(tempNature, tempFreq));
        Term term = new Term(sb.toString(), offe, termNatures);
        term.selfScore(-1 * tempFreq);
        TermUtil.insertTerm(terms, term, type);
    }

    /**
     * 重置
     */
    private void reset() {
        offe = -1;
        endOffe = -1;
        tempFreq = 50;
        tempNature = null;
        branch = forest;
    }

    /**
     * 传入一个term 返回这个term的状态
     * 
     * @param branch
     * @param term
     * @return
     */
    private SmartForest<String[]> termStatus(SmartForest<String[]> branch, Term term) {
        String name = term.getName();
        SmartForest<String[]> sf = branch;
        for (int j = 0; j < name.length(); j++) {
            sf = sf.get(name.charAt(j));
            if (sf == null) {
                return null;
            }
        }
        return sf;
    }

}
