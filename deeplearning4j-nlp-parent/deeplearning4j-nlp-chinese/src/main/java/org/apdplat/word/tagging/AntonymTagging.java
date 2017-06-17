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

package org.apdplat.word.tagging;

import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.segmentation.Word;
import org.apdplat.word.util.AutoDetector;
import org.apdplat.word.util.GenericTrie;
import org.apdplat.word.util.ResourceLoader;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * 反义标注
 * @author 杨尚川
 */
public class AntonymTagging {
    private AntonymTagging(){}

    private static final Logger LOGGER = LoggerFactory.getLogger(AntonymTagging.class);
    private static final GenericTrie<String[]> GENERIC_TRIE = new GenericTrie<>();
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
                LOGGER.info("初始化Antonym");
                int count = 0;
                for (String line : lines) {
                    try {
                        String[] words = line.split("\\s+");
                        if (words != null && words.length > 1) {
                            addWords(words);
                            count++;
                        } else {
                            LOGGER.error("错误的Antonym数据：" + line);
                        }
                    } catch (Exception e) {
                        LOGGER.error("错误的Antonym数据：" + line);
                    }
                }
                LOGGER.info("Antonym初始化完毕，数据条数：" + count);
            }

            @Override
            public void add(String line) {
                try {
                    String[] words = line.split("\\s+");
                    if (words != null && words.length > 1) {
                        addWords(words);
                    } else {
                        LOGGER.error("错误的Antonym数据：" + line);
                    }
                } catch (Exception e) {
                    LOGGER.error("错误的Antonym数据：" + line);
                }
            }

            @Override
            public void remove(String line) {
                try {
                    String[] words = line.split("\\s+");
                    if (words != null && words.length > 1) {
                        for (String word : words) {
                            GENERIC_TRIE.remove(word.trim());
                        }
                    } else {
                        LOGGER.error("错误的Antonym数据：" + line);
                    }
                } catch (Exception e) {
                    LOGGER.error("错误的Antonym数据：" + line);
                }
            }

            private void addWords(String[] words) {
                String word = words[0];
                String[] antonym = Arrays.copyOfRange(words, 1, words.length);
                //正向
                addAntonym(word, antonym);
                //反向
                for (String item : antonym) {
                    addAntonym(item, word);
                }
            }

            private void addAntonym(String word, String... words) {
                String[] exist = GENERIC_TRIE.get(word);
                if (exist != null) {
                    if(LOGGER.isDebugEnabled()) {
                        LOGGER.debug(word + " 已经有存在的反义词：");
                        for (String e : exist) {
                            LOGGER.debug("\t" + e);
                        }
                    }
                    Set<String> set = new HashSet<>();
                    set.addAll(Arrays.asList(exist));
                    set.addAll(Arrays.asList(words));
                    String[] merge = set.toArray(new String[0]);
                    if(LOGGER.isDebugEnabled()) {
                        LOGGER.debug("合并新的反义词：");
                        for (String e : words) {
                            LOGGER.debug("\t" + e);
                        }
                        LOGGER.debug("合并结果：");
                        for (String e : merge) {
                            LOGGER.debug("\t" + e);
                        }
                    }
                    GENERIC_TRIE.put(word.trim(), merge);
                } else {
                    GENERIC_TRIE.put(word.trim(), words);
                }
            }
        }, WordConfTools.get("word.antonym.path", "classpath:word_antonym.txt"));
    }
    public static void process(List<Word> words){
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("对分词结果进行反义标注之前：{}", words);
        }
        //反义并行标注
        words.parallelStream().forEach(word -> process(word));
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("对分词结果进行反义标注之后：{}", words);
        }
    }
    private static void process(Word word){
        String[] antonym = GENERIC_TRIE.get(word.getText());
        if(antonym!=null && antonym.length>1){
            //有反义词
            List<Word> antonymList = toWord(antonym);
            word.setAntonym(antonymList);
        }
    }
    private static List<Word> toWord(String[] words){
        List<Word> result = new ArrayList<>(words.length);
        for (String word : words){
            result.add(new Word(word));
        }
        return result;
    }

    public static void main(String[] args) {
        List<Word> words = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMaximumMatching).seg("5月初有哪些电影值得观看");
        System.out.println(words);
        AntonymTagging.process(words);
        System.out.println(words);
        words = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMaximumMatching).seg("由于工作不到位、服务不完善导致顾客在用餐时发生不愉快的事情,餐厅方面应该向顾客作出真诚的道歉,而不是敷衍了事。");
        System.out.println(words);
        AntonymTagging.process(words);
        System.out.println(words);
    }
}
