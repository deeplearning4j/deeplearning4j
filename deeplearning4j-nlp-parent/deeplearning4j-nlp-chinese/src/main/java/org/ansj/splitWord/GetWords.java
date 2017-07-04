package org.ansj.splitWord;

public interface GetWords {
    /**
     * 全文全词全匹配
     * 
     * @param str
     *            传入的需要分词的句子
     * @return 返还分完词后的句子
     */
    public String allWords();

    /**
     * 同一个对象传入词语
     * 
     * @param temp
     *            传入的句子
     */
    public void setStr(String temp);

    /**
     * 
     * @return
     */

    public void setChars(char[] chars, int start, int end);

    public int getOffe();
}
