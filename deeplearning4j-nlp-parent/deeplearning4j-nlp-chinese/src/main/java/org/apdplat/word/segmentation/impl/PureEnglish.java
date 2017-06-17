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

package org.apdplat.word.segmentation.impl;

import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.Word;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * 针对纯英文文本的分词器
 * @author 杨尚川
 */
public class PureEnglish implements Segmentation {
    private static final Logger LOGGER = LoggerFactory.getLogger(PureEnglish.class);
    private static final Pattern NUMBER = Pattern.compile("\\d+");
    private static final Pattern UNICODE = Pattern.compile("[uU][0-9a-fA-F]{4}");

    @Override
    public SegmentationAlgorithm getSegmentationAlgorithm() {
        return SegmentationAlgorithm.PureEnglish;
    }

    @Override
    public List<Word> seg(String text) {
        List<Word> segResult = new ArrayList<>();
        //以非字母数字字符切分行
        String[] words = text.trim().split("[^a-zA-Z0-9]");
        for (String word : words) {
            if ("".equals(word) || word.length()<2) {
                continue;
            }
            List<String> list = new ArrayList<>();
            //转换为全部小写
            if (word.length() < 6
                    //PostgreSQL等
                    || (Character.isUpperCase(word.charAt(word.length()-1))
                    && Character.isUpperCase(word.charAt(0)))
                    //P2P,Neo4j等
                    || NUMBER.matcher(word).find()
                    || isAllUpperCase(word)) {
                word = word.toLowerCase();
            }
            //按照大写字母进行单词拆分
            int last = 0;
            for (int i = 1; i < word.length(); i++) {
                if (Character.isUpperCase(word.charAt(i))
                        && Character.isLowerCase(word.charAt(i - 1))) {
                    list.add(word.substring(last, i));
                    last = i;
                }
            }
            if (last < word.length()) {
                list.add(word.substring(last, word.length()));
            }
            list.stream()
                    .map(w -> w.toLowerCase())
                    .forEach(w -> {
                        if (w.length() < 2) {
                            return;
                        }
                        w = irregularity(w);
                        if(w != null) {
                            segResult.add(new Word(w));
                        }
                    });
        }
        return segResult;
    }

    /**
     * 处理分词意外，即无规则情况
     * @param word
     * @return
     */
    private static String irregularity(String word){
        if(Character.isDigit(word.charAt(0))){
            LOGGER.debug("词以数字开头，忽略："+word);
            return null;
        }
        if(word.startsWith("0x")
                || word.startsWith("0X")){
            LOGGER.debug("词为16进制，忽略："+word);
            return null;
        }
        if(word.endsWith("l")
                && isNumeric(word.substring(0, word.length()-1))){
            LOGGER.debug("词为long类型数字，忽略："+word);
            return null;
        }
        if(UNICODE.matcher(word).find()){
            LOGGER.debug("词为UNICODE字符编码，忽略："+word);
            return null;
        }
        switch (word){
            //I’ll do it. You'll see.
            case "ll": return "will";
            //If you’re already building applications using Spring.
            case "re": return "are";
            //package com.manning.sdmia.ch04;
            case "ch": return "chapter";
            //you find you’ve made a
            case "ve": return "have";
            //but it doesn’t stop there.
            case "doesn": return "does";
            //but it isn’t enough.
            case "isn": return "is";
            //<input type="text" name="firstName" /><br/>
            case "br": return null;
        }
        return word;
    }

    private boolean isAllUpperCase(String string) {
        for(char c : string.toCharArray()){
            if(Character.isLowerCase(c)){
                return false;
            }
        }
        return true;
    }

    private static boolean isNumeric(String string) {
        for(char c : string.toCharArray()){
            if(!Character.isDigit(c)){
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        Segmentation segmentation = new PureEnglish();
        System.out.println(segmentation.seg("Your function may also be added permanently to Hive, however this requires a small modification to a Hive Java file and then rebuilding Hive."));
    }
}
