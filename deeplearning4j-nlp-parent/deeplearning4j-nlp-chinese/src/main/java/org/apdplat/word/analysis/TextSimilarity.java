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

import org.apdplat.word.recognition.StopWord;
import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.segmentation.Word;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 文本相似度
 * @author 杨尚川
 */
public abstract class TextSimilarity implements Similarity, SimilarityRanker{
    protected static final Logger LOGGER = LoggerFactory.getLogger(TextSimilarity.class);

    //默认分词器
    private Segmentation segmentation = null;
    //是否忽略停用词
    protected boolean filterStopWord = false;

    public void setSegmentationAlgorithm(SegmentationAlgorithm segmentationAlgorithm){
        segmentation = SegmentationFactory.getSegmentation(segmentationAlgorithm);
        LOGGER.info("设置分词算法为："+segmentationAlgorithm.getDes());
    }
    /**
     * 文本1和文本2的相似度分值
     * @param text1 文本1
     * @param text2 文本2
     * @return 相似度分值
     */
    @Override
    public double similarScore(String text1, String text2) {
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("文本1：");
            LOGGER.debug("\t" + text1);
            LOGGER.debug("文本2：");
            LOGGER.debug("\t" + text2);
        }
        if(text1 == null || text2 == null){
            //只要有一个文本为null，规定相似度分值为0，表示完全不相等
            return 0.0;
        }
        //分词
        List<Word> words1 = seg(text1);
        List<Word> words2 = seg(text2);
        //计算相似度分值
        return similarScore(words1, words2);
    }

    /**
     * 词列表1和词列表2的相似度分值
     * @param words1 词列表1
     * @param words2 词列表2
     * @return 相似度分值
     */
    @Override
    public double similarScore(List<Word> words1, List<Word> words2) {
        if(words1 == null || words2 == null){
            //只要有一个文本为null，规定相似度分值为0，表示完全不相等
            return 0.0;
        }
        if(words1.isEmpty() && words2.isEmpty()){
            //如果两个文本都为空，规定相似度分值为1，表示完全相等
            return 1.0;
        }
        if(words1.isEmpty() || words2.isEmpty()){
            //如果一个文本为空，另一个不为空，规定相似度分值为0，表示完全不相等
            return 0.0;
        }
        //输出词列表信息
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("词列表1：");
            LOGGER.debug("\t" + words1);
            LOGGER.debug("词列表2：");
            LOGGER.debug("\t" + words2);
        }
        double score = scoreImpl(words1, words2);
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("分值："+score);
        }
        score = (int)(score*1000000+0.5)/(double)1000000;
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("取六位小数，四舍五入，分值："+score);
        }
        return score;
    }

    /**
     * 计算相似度分值
     * @param words1 词列表1
     * @param words2 词列表2
     * @return 相似度分值
     */
    protected abstract double scoreImpl(List<Word> words1, List<Word> words2);

    /**
     * 对文本进行分词
     * @param text 文本
     * @return 分词结果
     */
    private List<Word> seg(String text){
        if(text == null){
            return Collections.emptyList();
        }
        if(segmentation == null){
            //延迟初始化
            segmentation = SegmentationFactory.getSegmentation(SegmentationAlgorithm.MaxNgramScore);
        }
        List<Word> words = segmentation.seg(text);
        if(filterStopWord) {
            //停用词过滤
            StopWord.filterStopWords(words);
        }
        return words;
    }

    /**
     * 如果没有指定权重，则默认使用词频来标注词的权重
     * 词频数据怎么来？
     * 一个词在词列表1中出现了几次，它在词列表1中的权重就是几
     * 一个词在词列表2中出现了几次，它在词列表2中的权重就是几
     * 标注好的权重存储在Word类的weight字段中
     * @param words1 词列表1
     * @param words2 词列表2
     */
    protected void taggingWeightWithWordFrequency(List<Word> words1, List<Word> words2){
        if(words1.get(0).getWeight() != null || words2.get(0).getWeight() != null){
            if(LOGGER.isDebugEnabled()){
                LOGGER.debug("词已经被指定权重，不再使用词频进行标注");
            }
            return;
        }
        //词频统计
        Map<String, AtomicInteger> frequency1 = frequency(words1);
        Map<String, AtomicInteger> frequency2 = frequency(words2);
        //输出词频统计信息
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("词频统计1：\n{}", formatWordsFrequency(frequency1));
            LOGGER.debug("词频统计2：\n{}", formatWordsFrequency(frequency2));
        }
        //权重标注
        words1.parallelStream().forEach(word->{
            word.setWeight(frequency1.get(word.getText()).floatValue());
        });
        words2.parallelStream().forEach(word->{
            word.setWeight(frequency2.get(word.getText()).floatValue());
        });
    }

    /**
     * 构造权重快速搜索容器
     * @param words 词列表
     * @return Map
     */
    protected Map<String, Float> toFastSearchMap(List<Word> words){
        Map<String, Float> weights = new ConcurrentHashMap<>();
        if(words == null){
            return weights;
        }
        words.parallelStream().forEach(word -> {
            if(word.getWeight() != null) {
                weights.put(word.getText(), word.getWeight());
            }else{
                LOGGER.error("词没有权重信息："+word.getText());
            }
        });
        return weights;
    }

    /**
     * 统计词频
     * @param words 词列表
     * @return 词频统计结果
     */
    private Map<String, AtomicInteger> frequency(List<Word> words){
        Map<String, AtomicInteger> frequency =new HashMap<>();
        words.forEach(word->{
            frequency.computeIfAbsent(word.getText(), k -> new AtomicInteger()).incrementAndGet();
        });
        return frequency;
    }

    /**
     * 格式化词频统计信息
     * @param frequency 词频统计信息
     */
    private String formatWordsFrequency(Map<String, AtomicInteger> frequency){
        StringBuilder str = new StringBuilder();
        if(frequency != null && !frequency.isEmpty()) {
            AtomicInteger c = new AtomicInteger();
            frequency
                    .entrySet()
                    .stream()
                    .sorted((a, b) -> b.getValue().get() - a.getValue().get())
                    .forEach(e -> str.append("\t").append(c.incrementAndGet()).append("、").append(e.getKey()).append("=").append(e.getValue()).append("\n"));
        }
        str.setLength(str.length()-1);
        return str.toString();
    }
}
