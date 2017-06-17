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
 * 判定方式：编辑距离（Edit Distance）
 * 指两个字串之间，由一个转成另一个所需的最少编辑操作次数
 * 允许的编辑操作包括将一个字符替换成另一个字符，增加一个字符，删除一个字符
 * 例如将kitten一字转成sitting：
 * sitten （k→s）将一个字符k替换成另一个字符s
 * sittin （e→i）将一个字符e替换成另一个字符i
 * sitting （→g）增加一个字符g
 * 因为这个算法是俄罗斯科学家Vladimir Levenshtein在1965年提出
 * 所以编辑距离（Edit Distance）又称Levenshtein距离
 * @author 杨尚川
 */
public class EditDistanceTextSimilarity extends TextSimilarity {
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
        int maxTextLength = Math.max(text1.length(), text2.length());
        if(maxTextLength == 0){
            //两个空字符串
            return 1.0;
        }
        //计算文本1和文本2的编辑距离
        int editDistance = editDistance(text1.toString(), text2.toString());
        double score = (1 - editDistance / (double)maxTextLength);
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("文本1："+text1.toString());
            LOGGER.debug("文本2："+text2.toString());
            LOGGER.debug("文本1和文本2的编辑距离："+editDistance);
            LOGGER.debug("文本1和文本2的最大长度："+maxTextLength);
            LOGGER.debug("文本1和文本2的相似度分值：1 - "+editDistance+" / (double)"+maxTextLength+"="+score);
        }
        return score;
    }

    private int editDistance(String text1, String text2) {
        int[] costs = new int[text2.length() + 1];
        for (int i = 0; i <= text1.length(); i++) {
            int previousValue = i;
            for (int j = 0; j <= text2.length(); j++) {
                if (i == 0) {
                    costs[j] = j;
                }
                else if (j > 0) {
                    int useValue = costs[j - 1];
                    if (text1.charAt(i - 1) != text2.charAt(j - 1)) {
                        useValue = Math.min(Math.min(useValue, previousValue), costs[j]) + 1;
                    }
                    costs[j - 1] = previousValue;
                    previousValue = useValue;

                }
            }
            if (i > 0) {
                costs[text2.length()] = previousValue;
            }
        }
        return costs[text2.length()];
    }

    public static void main(String[] args) {
        String text1 = "我爱购物";
        String text2 = "我爱读书";
        String text3 = "他是黑客";
        TextSimilarity textSimilarity = new EditDistanceTextSimilarity();
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
