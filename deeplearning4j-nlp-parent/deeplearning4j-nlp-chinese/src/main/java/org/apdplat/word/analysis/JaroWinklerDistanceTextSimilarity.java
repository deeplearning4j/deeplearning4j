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
 * 判定方式：Jaro–Winkler距离（Jaro–Winkler Distance），Jaro的扩展
 * 由William E. Winkler提出，最适合计算短文本如人名的相似度
 * 这里需要注意的是Jaro–Winkler距离也就是相似度分值
 * @author 杨尚川
 */
public class JaroWinklerDistanceTextSimilarity extends JaroDistanceTextSimilarity {
    private static final double DEFAULT_SCALING_FACTOR = 0.1;
    private static final int MAXIMUM_CHARACTERS = 4;
    private double scalingFactor;

    public JaroWinklerDistanceTextSimilarity(){
        this.scalingFactor = DEFAULT_SCALING_FACTOR;
    }

    /**
     * scalingFactor的值介于闭区间[0, 0.25]
     * @param scalingFactor
     */
    public JaroWinklerDistanceTextSimilarity(double scalingFactor) {
        if (scalingFactor > 0.25) {
            scalingFactor = 0.25;
        }
        if(scalingFactor < 0){
            scalingFactor = 0;
        }
        this.scalingFactor = scalingFactor;
    }
    /**
     * 计算相似度分值
     * @param words1 词列表1
     * @param words2 词列表2
     * @return 相似度分值
     */
    @Override
    protected double scoreImpl(List<Word> words1, List<Word> words2){
        double score = super.scoreImpl(words1, words2);

        score += (scalingFactor * commonPrefixLength() * (1.0 - score));

        return score;
    }

    /**
     * 判断两段文本的共同前缀的字符个数，共同前缀的字符个数如果大于4则按4处理
     * @return 整数闭区间[0, 4]
     */
    private int commonPrefixLength() {
        // shorterText和longerText已经在父类JaroDistanceTextSimilarity中准备好了
        // 这里直接用就可以
        int result = 0;
        int len = shorterText.length();
        for (int i = 0; i < len; i++) {
            if (shorterText.charAt(i) != longerText.charAt(i)) {
                break;
            }
            result++;
            if(result >= MAXIMUM_CHARACTERS){
                // 最多4个字符即可
                return MAXIMUM_CHARACTERS;
            }
        }
        return result;
    }

    public static void main(String[] args) {
        String text1 = "我爱购物";
        String text2 = "我爱读书";
        String text3 = "他是黑客";
        TextSimilarity textSimilarity = new JaroWinklerDistanceTextSimilarity();
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
