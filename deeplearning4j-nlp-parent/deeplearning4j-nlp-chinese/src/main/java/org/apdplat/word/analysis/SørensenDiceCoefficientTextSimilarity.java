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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 文本相似度计算
 * 判定方式：Sørensen–Dice系数（Sørensen–Dice coefficient），通过计算两个集合交集的大小的2倍除以两个集合的大小之和来评估他们的相似度
 * 算法步骤描述：
 * 1、分词
 * 2、求交集（去重），计算交集的不重复词的个数 intersectionSize
 * 3、两个集合的大小分别为 set1Size 和 set2Size
 * 4、相似度分值 = 2*intersectionSize/(set1Size+set2Size)
 * 完整计算公式：
 * double score = 2*intersectionSize/(set1Size+set2Size);
 * @author 杨尚川
 */
public class SørensenDiceCoefficientTextSimilarity extends TextSimilarity {
    /**
     * 计算相似度分值
     * @param words1 词列表1
     * @param words2 词列表2
     * @return 相似度分值
     */
    @Override
    protected double scoreImpl(List<Word> words1, List<Word> words2) {
        if(words1.isEmpty() && words2.isEmpty()){
            return 1.0;
        }
        //转变为不重复的集合
        Set<Word> words1Set = new HashSet<>(words1);
        Set<Word> words2Set = new HashSet<>(words2);
        // 两个集合的大小
        int set1Size = words1Set.size();
        int set2Size = words2Set.size();

        // 求交集（去重），计算交集的不重复词的个数
        words1Set.retainAll(words2Set);
        int intersectionSize = words1Set.size();

        //相似度分值
        double score = 2*intersectionSize / (double)(set1Size+set2Size);
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("集合1：" + words1);
            LOGGER.debug("集合2：" + words2);
            LOGGER.debug("集合1的大小：" + set1Size);
            LOGGER.debug("集合2的大小：" + set2Size);
            LOGGER.debug("交集的大小：" + intersectionSize);
            LOGGER.debug("相似度分值=2*" + intersectionSize + "/(double)(" + set1Size + "+" + set2Size + ")=" + score);
        }
        return score;
    }

    public static void main(String[] args) {
        String text1 = "我爱购物";
        String text2 = "我爱读书";
        String text3 = "他是黑客";
        TextSimilarity textSimilarity = new SørensenDiceCoefficientTextSimilarity();
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
