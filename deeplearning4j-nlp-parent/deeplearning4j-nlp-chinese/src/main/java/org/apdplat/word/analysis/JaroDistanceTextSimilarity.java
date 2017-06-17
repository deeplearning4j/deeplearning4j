/**
 *
 * APDPlat - Application Product Development Platform
 * Copyright (c) 2013, 杨尚川, yang-shangchuan@qq.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

package org.apdplat.word.analysis;

import org.apdplat.word.segmentation.Word;

import java.util.List;

/**
 * 文本相似度计算
 * 判定方式：Jaro距离（Jaro Distance），编辑距离的一种类型
 * 这里需要注意的是Jaro距离也就是相似度分值
 * @author 杨尚川
 */
public class JaroDistanceTextSimilarity extends TextSimilarity {
    protected String shorterText = null;
    protected String longerText = null;
    /**
     * 计算相似度分值
     * @param words1 词列表1
     * @param words2 词列表2
     * @return 相似度分值
     */
    @Override
    protected double scoreImpl(List<Word> words1, List<Word> words2){
        //文本1
        StringBuilder text1 = new StringBuilder();
        words1.forEach(word -> text1.append(word.getText()));
        //文本2
        StringBuilder text2 = new StringBuilder();
        words2.forEach(word -> text2.append(word.getText()));
        //计算文本1和文本2的Jaro距离
        //Jaro距离也就是相似度分值
        double score = jaroDistance(text1.toString(), text2.toString());
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("文本1："+text1.toString());
            LOGGER.debug("文本2："+text2.toString());
            LOGGER.debug("文本1和文本2的相似度分值："+score);
        }
        return score;
    }

    private double jaroDistance(String text1, String text2) {
        //假设文本1长度更短
        shorterText = text1.toLowerCase();
        longerText = text2.toLowerCase();
        //如果假设不成立则交换变量的值
        if (shorterText.length() > longerText.length()) {
            String temp = shorterText;
            shorterText = longerText;
            longerText = temp;
        }
        //字符交集窗口大小
        int windowLength = (shorterText.length() / 2) - 1;
        //求字符交集，m1可能会不等于m2
        String m1 = getCharacterConjunction(shorterText, longerText, windowLength);
        String m2 = getCharacterConjunction(longerText, shorterText, windowLength);
        //一种或两种情况没有字符交集，完全不相关，相似度分值为0
        if (m1.length() == 0 || m2.length() == 0) {
            return 0.0;
        }
        //交集字符个数不相等，完全不相关，相似度分值为0
        if (m1.length() != m2.length()) {
            return 0.0;
        }
        //m is the number of matching characters
        //m1.length() == m2.length()
        int m = m1.length();
        //两段文本为了保持相等需要的换位次数
        int transpositions = transpositions(m1, m2);
        //换位次数除以2
        //t is half the number of transpositions
        int t = transpositions/2;;
        //计算距离（这里的距离也就是相似度分值了）
        double distance = ( m / (double)shorterText.length() +
                            m / (double)longerText.length()  +
                            (m - t) / (double)m ) / 3.0;
        return distance;
    }

    /**
     * 获取两段文本的共有字符即字符交集
     * @param text1 文本1
     * @param text2 文本2
     * @param windowLength 字符交集窗口大小
     * @return 字符交集
     */
    private String getCharacterConjunction(String text1, String text2, int windowLength) {
        StringBuilder conjunction = new StringBuilder();
        StringBuilder target = new StringBuilder(text2);
        int len1 = text1.length();
        for (int i = 0; i < len1; i++) {
            char source = text1.charAt(i);
            boolean found = false;
            int start = Math.max(0, i - windowLength);
            int end = Math.min(i + windowLength, text2.length());
            for (int j = start; !found && j < end; j++) {
                if (source == target.charAt(j)) {
                    found = true;
                    conjunction.append(source);
                    target.setCharAt(j,'*');
                }
            }
        }
        return conjunction.toString();
    }

    /**
     * 计算两段文本为了保持相等需要的换位次数
     * @param text1 文本1
     * @param text2 文本2
     * @return 换位次数
     */
    private int transpositions(String text1, String text2) {
        int transpositions = 0;
        for (int i = 0; i < text1.length(); i++) {
            if (text1.charAt(i) != text2.charAt(i)) {
                transpositions++;
            }
        }
        return transpositions;
    }

    public static void main(String[] args) {
        String text1 = "我爱购物";
        String text2 = "我爱读书";
        String text3 = "他是黑客";
        TextSimilarity textSimilarity = new JaroDistanceTextSimilarity();
        double score1pk1 = textSimilarity.similarScore(text1, text1);
        double score1pk2 = textSimilarity.similarScore(text1, text2);
        double score1pk3 = textSimilarity.similarScore(text1, text3);
        double score2pk2 = textSimilarity.similarScore(text2, text2);
        double score2pk3 = textSimilarity.similarScore(text2, text3);
        double score3pk3 = textSimilarity.similarScore(text3, text3);
        System.out.println(text1+" 和 "+text1+" 的相似度分值："+score1pk1);
        System.out.println(text1+" 和 "+text2+" 的相似度分值："+score1pk2);
        System.out.println(text1+" 和 "+text3+" 的相似度分值："+score1pk3);
        System.out.println(text2+" 和 "+text2+" 的相似度分值："+score2pk2);
        System.out.println(text2+" 和 "+text3+" 的相似度分值："+score2pk3);
        System.out.println(text3+" 和 "+text3+" 的相似度分值："+score3pk3);
    }
}
