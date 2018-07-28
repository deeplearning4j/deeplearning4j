package org.ansj.util;

import org.ansj.domain.AnsjItem;
import org.ansj.domain.Term;
import org.ansj.domain.TermNatures;
import org.ansj.library.DATDictionary;
import org.ansj.splitWord.Analysis.Merger;
import org.ansj.util.TermUtil.InsertTermType;

import java.util.List;
import java.util.Map;

/**
 * 最短路径
 * 
 * @author ansj
 * 
 */
public class Graph {
    public char[] chars = null;
    public String realStr = null;
    public Term[] terms = null;
    protected Term end = null;
    protected Term root = null;
    protected static final String E = "末##末";
    protected static final String B = "始##始";
    // 是否有人名
    public boolean hasPerson;
    // 是否有数字
    public boolean hasNum;

    // 是否需有歧异

    public Graph(String str) {
        realStr = str;
        this.chars = str.toCharArray();
        terms = new Term[chars.length + 1];
        end = new Term(E, chars.length, AnsjItem.END);
        root = new Term(B, -1, AnsjItem.BEGIN);
        terms[chars.length] = end;
    }

    /**
     * 构建最优路径
     */
    public List<Term> getResult(Merger merger) {
        return merger.merger();
    }

    /**
     * 增加一个词语到图中
     * 
     * @param term
     */
    public void addTerm(Term term) {
        // 是否有数字
        if (!hasNum && term.termNatures().numAttr.numFreq > 0) {
            hasNum = true;
        }
        // 是否有人名
        if (!hasPerson && term.termNatures().personAttr.flag) {
            hasPerson = true;
        }
        TermUtil.insertTerm(terms, term, InsertTermType.REPLACE);

    }

    /**
     * 取得最优路径的root Term
     * 
     * @return
     */
    protected Term optimalRoot() {
        Term to = end;
        to.clearScore();
        Term from = null;
        while ((from = to.from()) != null) {
            for (int i = from.getOffe() + 1; i < to.getOffe(); i++) {
                terms[i] = null;
            }
            if (from.getOffe() > -1) {
                terms[from.getOffe()] = from;
            }
            // 断开横向链表.节省内存
            from.setNext(null);
            from.setTo(to);
            from.clearScore();
            to = from;
        }
        return root;
    }

    /**
     * 删除最短的节点
     */
    public void rmLittlePath() {
        int maxTo = -1;
        Term temp = null;
        Term maxTerm = null;
        // 是否有交叉
        boolean flag = false;
        final int length = terms.length - 1;
        for (int i = 0; i < length; i++) {
            maxTerm = getMaxTerm(i);
            if (maxTerm == null)
                continue;

            maxTo = maxTerm.toValue();

            /**
             * 对字数进行优化.如果一个字.就跳过..两个字.且第二个为null则.也跳过.从第二个后开始
             */
            switch (maxTerm.getName().length()) {
                case 1:
                    continue;
                case 2:
                    if (terms[i + 1] == null) {
                        i = i + 1;
                        continue;
                    }
            }

            /**
             * 判断是否有交叉
             */
            for (int j = i + 1; j < maxTo; j++) {
                temp = getMaxTerm(j);
                if (temp == null) {
                    continue;
                }
                if (maxTo < temp.toValue()) {
                    maxTo = temp.toValue();
                    flag = true;
                }
            }

            if (flag) {
                i = maxTo - 1;
                flag = false;
            } else {
                maxTerm.setNext(null);
                terms[i] = maxTerm;
                for (int j = i + 1; j < maxTo; j++) {
                    terms[j] = null;
                }
                // FIXME: 这里理论上得设置。但是跑了这么久，还不发生错误。应该是不依赖于双向链接。需要确认下。这段代码是否有用
                // //将下面的to的from设置回来
                // temp = terms[i+maxTerm.getName().length()] ;
                // do{
                // temp.setFrom(maxTerm) ;
                // }while((temp=temp.next())!=null) ;

            }
        }
    }

    /**
     * 得道最到本行最大term,也就是最右面的term
     * 
     * @param i
     * @return
     */
    private Term getMaxTerm(int i) {
        Term maxTerm = terms[i];
        if (maxTerm == null) {
            return null;
        }
        Term term = maxTerm;
        while ((term = term.next()) != null) {
            maxTerm = term;
        }
        return maxTerm;
    }

    /**
     * 删除无意义的节点,防止viterbi太多
     */
    public void rmLittleSinglePath() {
        int maxTo = -1;
        Term temp = null;
        for (int i = 0; i < terms.length; i++) {
            if (terms[i] == null)
                continue;
            maxTo = terms[i].toValue();
            if (maxTo - i == 1 || i + 1 == terms.length)
                continue;
            for (int j = i; j < maxTo; j++) {
                temp = terms[j];
                if (temp != null && temp.toValue() <= maxTo && temp.getName().length() == 1) {
                    terms[j] = null;
                }
            }
        }
    }

    /**
     * 删除小节点。保证被删除的小节点的单个分数小于等于大节点的分数
     */
    public void rmLittlePathByScore() {
        int maxTo = -1;
        Term temp = null;
        for (int i = 0; i < terms.length; i++) {
            if (terms[i] == null) {
                continue;
            }
            Term maxTerm = null;
            double maxScore = 0;
            Term term = terms[i];
            // 找到自身分数对大最长的

            do {
                if (maxTerm == null || maxScore > term.score()) {
                    maxTerm = term;
                } else if (maxScore == term.score() && maxTerm.getName().length() < term.getName().length()) {
                    maxTerm = term;
                }

            } while ((term = term.next()) != null);
            term = maxTerm;
            do {
                maxTo = term.toValue();
                maxScore = term.score();
                if (maxTo - i == 1 || i + 1 == terms.length)
                    continue;
                boolean flag = true;// 可以删除
                out: for (int j = i; j < maxTo; j++) {
                    temp = terms[j];
                    if (temp == null) {
                        continue;
                    }
                    do {
                        if (temp.toValue() > maxTo || temp.score() < maxScore) {
                            flag = false;
                            break out;
                        }
                    } while ((temp = temp.next()) != null);
                }
                // 验证通过可以删除了
                if (flag) {
                    for (int j = i + 1; j < maxTo; j++) {
                        terms[j] = null;
                    }
                }
            } while ((term = term.next()) != null);
        }
    }

    public void walkPathByScore() {
        Term term = null;
        // BEGIN先行打分
        mergerByScore(root, 0);
        // 从第一个词开始往后打分
        for (int i = 0; i < terms.length; i++) {
            term = terms[i];
            while (term != null && term.from() != null && term != end) {
                int to = term.toValue();
                mergerByScore(term, to);
                term = term.next();
            }
        }
        optimalRoot();
    }

    public void walkPath() {
        walkPath(null);
    }

    /**
     * 干涉性增加相对权重
     * 
     * @param relationMap
     */
    public void walkPath(Map<String, Double> relationMap) {
        Term term = null;
        // BEGIN先行打分
        merger(root, 0, relationMap);
        // 从第一个词开始往后打分
        for (int i = 0; i < terms.length; i++) {
            term = terms[i];
            while (term != null && term.from() != null && term != end) {
                int to = term.toValue();
                merger(term, to, relationMap);
                term = term.next();
            }
        }
        optimalRoot();
    }

    /**
     * 具体的遍历打分方法
     * 
     * @param i 起始位置
     * @param j 起始属性
     * @param to
     */
    private void merger(Term fromTerm, int to, Map<String, Double> relationMap) {
        Term term = null;
        if (terms[to] != null) {
            term = terms[to];
            while (term != null) {
                // 关系式to.set(from)
                term.setPathScore(fromTerm, relationMap);
                term = term.next();
            }
        } else {
            char c = chars[to];
            TermNatures tn = DATDictionary.getItem(c).termNatures;
            if (tn == null || tn == TermNatures.NULL) {
                tn = TermNatures.NULL;
            }
            terms[to] = new Term(String.valueOf(c), to, tn);
            terms[to].setPathScore(fromTerm, relationMap);
        }
    }

    /**
     * 根据分数
     * 
     * @param i 起始位置
     * @param j 起始属性
     * @param to
     */
    private void mergerByScore(Term fromTerm, int to) {
        Term term = null;
        if (terms[to] != null) {
            term = terms[to];
            while (term != null) {
                // 关系式to.set(from)
                term.setPathSelfScore(fromTerm);
                term = term.next();
            }
        }

    }

    /**
     * 对graph进行调试用的
     */
    public void printGraph() {
        for (Term term : terms) {
            if (term == null) {
                continue;
            }
            System.out.print(term.getName() + "\t" + term.score() + " ,");
            while ((term = term.next()) != null) {
                System.out.print(term + "\t" + term.score() + " ,");
            }
            System.out.println();
        }
    }

}
