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

package org.apdplat.word.dictionary;

import org.apdplat.word.recognition.RecognitionTool;
import org.apdplat.word.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * 词典工具
 * 1、把多个词典合并为一个并规范清理
 * 词长度：只保留大于等于2并且小于等于4的长度的词
 * 识别功能： 移除能识别的词
 * 移除非中文词：防止大量无意义或特殊词混入词典
 * 2、移除词典中的短语结构
 * @author 杨尚川
 */
public class DictionaryTools {    
    private static final Logger LOGGER = LoggerFactory.getLogger(DictionaryTools.class);
    public static void main(String[] args) throws IOException{
        List<String> sources = new ArrayList<>();
        sources.add("src/main/resources/dic.txt");
        sources.add("target/dic.txt");
        String target = "src/main/resources/dic.txt";
        merge(sources, target);
    }    
    /**
     * 移除词典中的短语结构
     * @param phrasePath
     * @param dicPath
     */
    public static void removePhraseFromDic(String phrasePath, String dicPath) {
        try(Stream<String> phrases = Files.lines(Paths.get(phrasePath), Charset.forName("utf-8"));
            Stream<String> words = Files.lines(Paths.get(dicPath), Charset.forName("utf-8"))){
            Set<String> set = new HashSet<>();
            phrases.forEach(phrase -> {
                String[] attr = phrase.split("=");
                if (attr != null && attr.length == 2) {
                    set.add(attr[0]);
                }
            });
            List<String> list = new ArrayList<>();
            AtomicInteger len = new AtomicInteger();
            words.forEach(word -> {
                len.incrementAndGet();
                if(!set.contains(word)){
                    list.add(word);
                }
            });
            set.clear();
            Files.write(Paths.get(dicPath), list, Charset.forName("utf-8"));
            LOGGER.info("移除短语结构数目："+(len.get() - list.size()));
        }catch(Exception e){
            LOGGER.error("移除短语结构失败：", e);
        }
    }
    /**
     * 把多个词典合并为一个
     * @param sources 多个待词典
     * @param target 合并后的词典
     * @throws IOException 
     */
    public static void merge(List<String> sources, String target) throws IOException{
        //读取所有需要合并的词典
        Set<String> set = new HashSet<>();
        AtomicInteger i = new AtomicInteger();
        for(String source : sources){
            try(Stream<String> lines = Files.lines(Paths.get(source), Charset.forName("utf-8"))){
                lines.forEach(line -> {
                    i.incrementAndGet();
                    line = line.trim();
                    // 词长度：大于等于2并且小于等于4
                    // 识别功能 能识别的词 就不用放到词典中了，没必要多此一举
                    //至少要两个中文字符，防止大量无意义或特殊词混入词典
                    if (line.length() > 4
                            || line.length() < 2
                            || !Utils.isChineseCharAndLengthAtLeastTwo(line)
                            || RecognitionTool.recog(line)) {
                        LOGGER.debug("过滤：" + line);
                        return;
                    }
                    set.add(line);
                });
            }
        }
        LOGGER.info("合并词数："+i.get());
        LOGGER.info("保留词数："+set.size());
        List<String> list = set.stream().sorted().collect(Collectors.toList());
        Files.write(Paths.get(target), list, Charset.forName("utf-8"));
    }
}
