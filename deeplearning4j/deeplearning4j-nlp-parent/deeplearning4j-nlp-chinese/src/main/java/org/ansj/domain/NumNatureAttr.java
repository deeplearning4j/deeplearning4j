package org.ansj.domain;

import java.io.Serializable;

public class NumNatureAttr implements Serializable {

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    public static final NumNatureAttr NULL = new NumNatureAttr();

    // 是有可能是一个数字
    public int numFreq = -1;

    // 数字的结尾
    public int numEndFreq = -1;

    // 最大词性是否是数字
    public boolean flag = false;

    public NumNatureAttr() {}
}
