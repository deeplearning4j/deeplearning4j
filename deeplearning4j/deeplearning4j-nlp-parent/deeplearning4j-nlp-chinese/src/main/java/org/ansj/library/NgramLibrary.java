package org.ansj.library;

import org.ansj.domain.Term;
import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.util.logging.LogFactory;

/**
 * 两个词之间的关联
 * 
 * @author ansj
 * 
 */
public class NgramLibrary {
    static {
        long start = System.currentTimeMillis();
        MyStaticValue.initBigramTables();
        LogFactory.getLog(NgramLibrary.class).info("init ngram ok use time :" + (System.currentTimeMillis() - start));
    }

    /**
     * 查找两个词与词之间的频率
     * 
     * @param from
     * @param to
     * @return
     */
    public static int getTwoWordFreq(Term from, Term to) {
        if (from.item().bigramEntryMap == null) {
            return 0;
        }
        Integer freq = from.item().bigramEntryMap.get(to.item().getIndex());
        if (freq == null) {
            return 0;
        } else {
            return freq;
        }
    }

}
