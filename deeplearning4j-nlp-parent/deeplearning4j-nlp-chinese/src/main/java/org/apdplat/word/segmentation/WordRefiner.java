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

package org.apdplat.word.segmentation;

import org.apdplat.word.util.AutoDetector;
import org.apdplat.word.util.GenericTrie;
import org.apdplat.word.util.ResourceLoader;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * 对分词结果进行微调
 * @author 杨尚川
 */
public class WordRefiner {
    private WordRefiner(){}

    private static final Logger LOGGER = LoggerFactory.getLogger(WordRefiner.class);
    private static final GenericTrie<String> GENERIC_TRIE = new GenericTrie<>();
    static{
        reload();
    }
    public static void reload(){
        AutoDetector.loadAndWatch(new ResourceLoader() {

            @Override
            public void clear() {
                GENERIC_TRIE.clear();
            }

            @Override
            public void load(List<String> lines) {
                LOGGER.info("初始化WordRefiner");
                int count = 0;
                for (String line : lines) {
                    try {
                        String[] attr = line.split("=");
                        GENERIC_TRIE.put(attr[0].trim(), attr[1].trim().replaceAll("\\s+", " "));
                        count++;
                    } catch (Exception e) {
                        LOGGER.error("错误的WordRefiner数据：" + line);
                    }
                }
                LOGGER.info("WordRefiner初始化完毕，数据条数：" + count);
            }

            @Override
            public void add(String line) {
                try {
                    String[] attr = line.split("=");
                    GENERIC_TRIE.put(attr[0].trim(), attr[1].trim().replaceAll("\\s+", " "));
                } catch (Exception e) {
                    LOGGER.error("错误的WordRefiner数据：" + line);
                }
            }

            @Override
            public void remove(String line) {
                try {
                    String[] attr = line.split("=");
                    GENERIC_TRIE.remove(attr[0].trim());
                } catch (Exception e) {
                    LOGGER.error("错误的WordRefiner数据：" + line);
                }
            }

        }, WordConfTools.get("word.refine.path", "classpath:word_refine.txt"));
    }
    /**
     * 将一个词拆分成几个，返回null表示不能拆分
     * @param word
     * @return
     */
    public static List<Word> split(Word word){
        String value = GENERIC_TRIE.get(word.getText());
        if(value==null){
            return null;
        }
        List<Word> words = new ArrayList<>();
        for(String val : value.split("\\s+")){
            words.add(new Word(val));
        }
        if(words.isEmpty()){
            return null;
        }
        return words;
    }

    /**
     * 将多个词合并成一个，返回null表示不能合并
     * @param words
     * @return
     */
    public static Word combine(List<Word> words){
        if(words==null || words.size() < 2){
            return null;
        }
        String key = "";
        for(Word word : words){
            key += word.getText();
            key += " ";
        }
        key=key.trim();
        String value = GENERIC_TRIE.get(key);
        if(value==null){
            return null;
        }
        return new Word(value);
    }

    /**
     * 先拆词，再组词
     * @param words
     * @return
     */
    public static List<Word> refine(List<Word> words){
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("对分词结果进行refine之前：{}", words);
        }
        List<Word> result = new ArrayList<>(words.size());
        //一：拆词
        for(Word word : words){
            List<Word> splitWords = WordRefiner.split(word);
            if(splitWords==null){
                result.add(word);
            }else{
                if(LOGGER.isDebugEnabled()) {
                    LOGGER.debug("词： " + word.getText() + " 被拆分为：" + splitWords);
                }
                result.addAll(splitWords);
            }
        }
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("对分词结果进行refine阶段的拆词之后：{}", result);
        }
        //二：组词
        if(result.size()<2){
            return result;
        }
        int combineMaxLength = WordConfTools.getInt("word.refine.combine.max.length", 3);
        if(combineMaxLength < 2){
            combineMaxLength = 2;
        }
        List<Word> finalResult = new ArrayList<>(result.size());
        for(int i=0; i<result.size(); i++){
            List<Word> toCombineWords = null;
            Word combinedWord = null;
            for(int j=2; j<=combineMaxLength; j++){
                int to = i+j;
                if(to > result.size()){
                    to = result.size();
                }
                toCombineWords = result.subList(i, to);
                combinedWord = WordRefiner.combine(toCombineWords);
                if(combinedWord != null){
                    i += j;
                    i--;
                    break;
                }
            }
            if(combinedWord == null){
                finalResult.add(result.get(i));
            }else{
                if(LOGGER.isDebugEnabled()) {
                    LOGGER.debug("词： " + toCombineWords + " 被合并为：" + combinedWord);
                }
                finalResult.add(combinedWord);
            }
        }
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("对分词结果进行refine阶段的组词之后：{}", finalResult);
        }
        return finalResult;
    }

    public static void main(String[] args) {
        List<Word> words = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMaximumMatching).seg("我国工人阶级和广大劳动群众要更加紧密地团结在党中央周围");
        System.out.println(words);
        words = WordRefiner.refine(words);
        System.out.println(words);
        words = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMaximumMatching).seg("在实现“两个一百年”奋斗目标的伟大征程上再创新的业绩");
        System.out.println(words);
        words = WordRefiner.refine(words);
        System.out.println(words);
    }
}
