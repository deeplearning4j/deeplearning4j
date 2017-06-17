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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigInteger;
import java.util.List;

/**
 * 文本相似度计算
 * 判定方式：SimHash + 汉明距离（Hamming Distance）
 * 先使用SimHash把不同长度的文本映射为等长文本，然后再计算等长文本的汉明距离
 *
 * simhash和普通hash最大的不同在于:
 * 普通hash对 仅有一个字节不同的文本 会映射成 两个完全不同的哈希结果
 * simhash对 相似的文本 会映射成 相似的哈希结果
 *
 * 汉明距离是以美国数学家Richard Wesley Hamming的名字命名的
 * 两个等长字符串之间的汉明距离是两个字符串相应位置的不同字符的个数
 * 换句话说，它就是将一个字符串变换成另外一个字符串所需要替换的字符个数
 *
 * 比如：
 * 1011101 与 1001001 之间的汉明距离是 2
 * 2143896 与 2233796 之间的汉明距离是 3
 * toned 与 roses 之间的汉明距离是 3
 * @author 杨尚川
 */
public class SimHashPlusHammingDistanceTextSimilarity extends TextSimilarity {
    private static final Logger LOGGER = LoggerFactory.getLogger(SimHashPlusHammingDistanceTextSimilarity.class);

    private int hashBitCount = 128;

    public SimHashPlusHammingDistanceTextSimilarity(){}

    public SimHashPlusHammingDistanceTextSimilarity(int hashBitCount) {
        this.hashBitCount = hashBitCount;
    }

    public int getHashBitCount() {
        return hashBitCount;
    }

    public void setHashBitCount(int hashBitCount) {
        this.hashBitCount = hashBitCount;
    }

    /**
     * 计算相似度分值
     * @param words1 词列表1
     * @param words2 词列表2
     * @return 相似度分值
     */
    @Override
    protected double scoreImpl(List<Word> words1, List<Word> words2){
        //用词频来标注词的权重
        taggingWeightWithWordFrequency(words1, words2);
        //计算SimHash
        String simHash1 = simHash(words1);
        String simHash2 = simHash(words2);
        //计算SimHash值之间的汉明距离
        int hammingDistance = hammingDistance(simHash1, simHash2);
        if(hammingDistance == -1){
            LOGGER.error("文本1：" + words1.toString());
            LOGGER.error("文本2：" + words2.toString());
            LOGGER.error("文本1SimHash值：" + simHash1);
            LOGGER.error("文本2SimHash值：" + simHash2);
            LOGGER.error("文本1和文本2的SimHash值长度不相等，不能计算汉明距离");
            return 0.0;
        }
        int maxDistance = simHash1.length();
        double score = (1 - hammingDistance / (double)maxDistance);
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("文本1：" + words1.toString());
            LOGGER.debug("文本2：" + words2.toString());
            LOGGER.debug("文本1SimHash值："+simHash1);
            LOGGER.debug("文本2SimHash值："+simHash2);
            LOGGER.debug("hashBitCount："+hashBitCount);
            LOGGER.debug("SimHash值之间的汉明距离："+hammingDistance);
            LOGGER.debug("文本1和文本2的相似度分值：1 - "+hammingDistance+" / (double)"+maxDistance+"="+score);
        }
        return score;
    }

    /**
     * 计算词列表的SimHash值
     * @param words 词列表
     * @return SimHash值
     */
    private String simHash(List<Word> words) {
        float[] hashBit = new float[hashBitCount];
        words.forEach(word -> {
            float weight = word.getWeight()==null?1:word.getWeight();
            BigInteger hash = hash(word.getText());
            for (int i = 0; i < hashBitCount; i++) {
                BigInteger bitMask = new BigInteger("1").shiftLeft(i);
                if (hash.and(bitMask).signum() != 0) {
                    hashBit[i] += weight;
                } else {
                    hashBit[i] -= weight;
                }
            }
        });
        StringBuffer fingerprint = new StringBuffer();
        for (int i = 0; i < hashBitCount; i++) {
            if (hashBit[i] >= 0) {
                fingerprint.append("1");
            }else{
                fingerprint.append("0");
            }
        }
        return fingerprint.toString();
    }

    /**
     * 计算词的哈希值
     * @param word 词
     * @return 哈希值
     */
    private BigInteger hash(String word) {
        if (word == null || word.length() == 0) {
            return new BigInteger("0");
        }
        char[] charArray = word.toCharArray();
        BigInteger x = BigInteger.valueOf(((long) charArray[0]) << 7);
        BigInteger m = new BigInteger("1000003");
        BigInteger mask = new BigInteger("2").pow(hashBitCount).subtract(new BigInteger("1"));
        long sum = 0;
        for (char c : charArray) {
            sum += c;
        }
        x = x.multiply(m).xor(BigInteger.valueOf(sum)).and(mask);
        x = x.xor(new BigInteger(String.valueOf(word.length())));
        if (x.equals(new BigInteger("-1"))) {
            x = new BigInteger("-2");
        }
        return x;
    }

    /**
     * 计算等长的SimHash值的汉明距离
     * 如不能比较距离（比较的两段文本长度不相等），则返回-1
     * @param simHash1 SimHash值1
     * @param simHash2 SimHash值2
     * @return 汉明距离
     */
    private int hammingDistance(String simHash1, String simHash2) {
        if (simHash1.length() != simHash2.length()) {
            return -1;
        }
        int distance = 0;
        int len = simHash1.length();
        for (int i = 0; i < len; i++) {
            if (simHash1.charAt(i) != simHash2.charAt(i)) {
                distance++;
            }
        }
        return distance;
    }

    public static void main(String[] args) throws Exception{
        String text1 = "我爱购物";
        String text2 = "我爱读书";
        String text3 = "他是黑客";
        TextSimilarity textSimilarity = new SimHashPlusHammingDistanceTextSimilarity();
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
