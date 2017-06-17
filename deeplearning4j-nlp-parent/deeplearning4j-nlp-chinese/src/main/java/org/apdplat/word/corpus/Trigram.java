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

package org.apdplat.word.corpus;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apdplat.word.segmentation.Word;
import org.apdplat.word.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 三元语法模型
 * @author 杨尚川
 */
public class Trigram {
    private static final Logger LOGGER = LoggerFactory.getLogger(Trigram.class);
    private static final DoubleArrayGenericTrie DOUBLE_ARRAY_GENERIC_TRIE = new DoubleArrayGenericTrie(WordConfTools.getInt("trigram.double.array.trie.size", 9800000));
    private static int maxFrequency = 0;
    static{
        reload();
    }
    public static void reload(){
        AutoDetector.loadAndWatch(new ResourceLoader(){

            @Override
            public void clear() {
                DOUBLE_ARRAY_GENERIC_TRIE.clear();
            }

            @Override
            public void load(List<String> lines) {
                LOGGER.info("初始化trigram");
                Map<String, Integer> map = new HashMap<>();
                for(String line : lines){
                    try{
                        addLine(line, map);
                    }catch(Exception e){
                        LOGGER.error("错误的trigram数据："+line);
                    }
                }
                int size = map.size();
                DOUBLE_ARRAY_GENERIC_TRIE.putAll(map);
                LOGGER.info("trigram初始化完毕，trigram数据条数：" + size);
            }

            @Override
            public void add(String line) {
                throw new RuntimeException("not yet support menthod!");
            }

            private void addLine(String line, Map<String, Integer> map){
                String[] attr = line.split("\\s+");
                int frequency = Integer.parseInt(attr[1]);
                if(frequency > maxFrequency){
                    maxFrequency = frequency;
                }
                map.put(attr[0], frequency);
            }

            @Override
            public void remove(String line) {
                throw new RuntimeException("not yet support menthod!");
            }
        
        }, WordConfTools.get("trigram.path", "classpath:trigram.txt"));
    }

    public static int getMaxFrequency() {
        return maxFrequency;
    }

    /**
     * 一次性计算多种分词结果的三元模型分值
     * @param sentences 多种分词结果
     * @return 分词结果及其对应的分值
     */
    public static Map<List<Word>, Float> trigram(List<Word>... sentences){
        Map<List<Word>, Float> map = new HashMap<>();
        //计算多种分词结果的分值
        for(List<Word> sentence : sentences){
            if(map.get(sentence) != null){
                //相同的分词结果只计算一次分值
                continue;
            }
            float score=0;
            //计算其中一种分词结果的分值
            if(sentence.size() > 2){
                for(int i=0; i<sentence.size()-2; i++){
                    String first = sentence.get(i).getText();
                    String second = sentence.get(i+1).getText();
                    String third = sentence.get(i+2).getText();
                    float trigramScore = getScore(first, second, third);
                    if(trigramScore > 0){
                        score += trigramScore;
                    }
                }
            }
            map.put(sentence, score);
        }
        
        return map;
    }
    /**
     * 计算分词结果的三元模型分值
     * @param words 分词结果
     * @return 三元模型分值
     */
    public static float trigram(List<Word> words){
        if(words.size() > 2){
            float score=0;
            for(int i=0; i<words.size()-2; i++){
                score += Trigram.getScore(words.get(i).getText(), words.get(i+1).getText(), words.get(i+2).getText());
            }
            return score;
        }
        return 0;
    }
    /**
     * 获取三个词前后紧挨着同时出现在语料库中的分值
     * 分值被归一化了：
     * 完全没有出现分值为0
     * 出现频率最高的分值为1
     * @param first 第一个词
     * @param second 第二个词
     * @param third 第三个词
     * @return 同时出现的分值
     */
    public static float getScore(String first, String second, String third) {
        int frequency = getFrequency(first, second, third);
        float score = frequency/(float)maxFrequency;
        if(LOGGER.isDebugEnabled()) {
            if(score>0) {
                LOGGER.debug("三元模型 " + first + ":" + second + ":" + third + " 获得分值：" + score);
            }
        }
        return score;
    }

    public static int getFrequency(String first, String second, String third) {
        Integer value = DOUBLE_ARRAY_GENERIC_TRIE.get(first+":"+second+":"+third);
        if(value == null){
            return 0;
        }
        return value;
    }
}