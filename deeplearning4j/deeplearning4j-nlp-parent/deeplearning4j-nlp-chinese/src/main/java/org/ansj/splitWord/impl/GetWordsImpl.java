package org.ansj.splitWord.impl;

import org.ansj.domain.AnsjItem;
import org.ansj.library.DATDictionary;
import org.ansj.splitWord.GetWords;

public class GetWordsImpl implements GetWords {

    /**
     * offe : 当前词的偏移量
     */
    public int offe;

    /**
     * 构造方法，同时加载词典,传入词语相当于同时调用了setStr() ;
     */
    public GetWordsImpl(String str) {
        setStr(str);
    }

    /**
     * 构造方法，同时加载词典
     */
    public GetWordsImpl() {}

    int charsLength = 0;

    @Override
    public void setStr(String str) {
        setChars(str.toCharArray(), 0, str.length());
    }

    @Override
    public void setChars(char[] chars, int start, int end) {
        this.chars = chars;
        i = start;
        this.start = start;
        charsLength = end;
        checkValue = 0;
    }

    public char[] chars;
    private int charHashCode;
    private int start = 0;
    public int end = 0;
    private int baseValue = 0;
    private int checkValue = 0;
    private int tempBaseValue = 0;
    public int i = 0;
    private String str = null;

    @Override
    public String allWords() {
        for (; i < charsLength; i++) {
            charHashCode = chars[i];
            end++;
            switch (getStatement()) {
                case 0:
                    if (baseValue == chars[i]) {
                        str = String.valueOf(chars[i]);
                        offe = i;
                        start = ++i;
                        end = 0;
                        baseValue = 0;
                        tempBaseValue = baseValue;
                        return str;
                    } else {
                        int startCharStatus = DATDictionary.getItem(chars[start]).getStatus();
                        if (startCharStatus == 1) { //如果start的词的status为1，则将start设为i；否则start加1
                            start = i;
                            i--;
                            end = 0;
                            baseValue = 0;
                        } else {
                            i = start;
                            start++;
                            end = 0;
                            baseValue = 0;
                        }
                        break;
                    }
                case 2:
                    i++;
                    offe = start;
                    tempBaseValue = baseValue;
                    return DATDictionary.getItem(tempBaseValue).getName();
                case 3:
                    offe = start;
                    start++;
                    i = start;
                    end = 0;
                    tempBaseValue = baseValue;
                    baseValue = 0;
                    return DATDictionary.getItem(tempBaseValue).getName();
            }

        }
        end = 0;
        baseValue = 0;
        i = 0;
        return null;
    }

    /**
     * 根据用户传入的c得到单词的状态. 0.代表这个字不在词典中 1.继续 2.是个词但是还可以继续 3.停止已经是个词了
     * 
     * @param c
     * @return
     */
    private int getStatement() {
        checkValue = baseValue;
        baseValue = DATDictionary.getItem(checkValue).getBase() + charHashCode;
        if (baseValue < DATDictionary.arrayLength && (DATDictionary.getItem(baseValue).getCheck() == checkValue
                        || DATDictionary.getItem(baseValue).getCheck() == -1)) {
            return DATDictionary.getItem(baseValue).getStatus();
        }
        return 0;
    }

    public AnsjItem getItem() {
        return DATDictionary.getItem(tempBaseValue);
    }

    @Override
    public int getOffe() {
        return offe;
    }

}
