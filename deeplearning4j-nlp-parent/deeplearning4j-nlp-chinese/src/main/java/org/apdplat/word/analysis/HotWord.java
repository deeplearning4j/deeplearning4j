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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apdplat.word.recognition.Punctuation;
import org.apdplat.word.recognition.RecognitionTool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 利用NGRAM做热词分析
 * @author 杨尚川
 */
public class HotWord {
    private static final Logger LOGGER = LoggerFactory.getLogger(HotWord.class);
    public static Map<String, Integer> get(String text, int ngram){
        Map<String, Integer> map = new HashMap<>();
        //根据标点符号对文本进行分割
        //根据英文单词对文本进行分割
        List<String> sentences = new ArrayList<>();
        for(String sentence : Punctuation.seg(text, false)){
            LOGGER.debug("判断句子是否有英文单词：{}", sentence);
            int start=0;
            for(int i=0; i<sentence.length(); i++){
                if(RecognitionTool.isEnglish(sentence.charAt(i))){
                    if(i>1 && !RecognitionTool.isEnglish(sentence.charAt(i-1))){
                        sentences.add(sentence.substring(start,i));
                        start=i+1;
                    }else{
                        start++;
                    }
                }
                if(i==sentence.length()-1){
                    sentences.add(sentence.substring(start,i+1));
                }
            }
        }
        for(String sentence : sentences){
            LOGGER.debug("\n\n分析文本：{}", sentence);
            int len = sentence.length()-ngram+1;
            for(int i=0; i<len; i++){
                String word = sentence.substring(i, i+ngram);
                System.out.print(word+" ");
                Integer count = map.get(word);
                if(count == null){
                    count = 1;
                }else{
                    count++;
                }
                map.put(word, count);
            }
        }
        return map;
    }
    public static void main(String[] args){
        Map<String, Integer> map = get("目前在南京论之语有限责任公司，开发一部二组实习生，小孔同学，小孔同学是一名IT男。", 4);

        map.entrySet().stream().sorted((a,b)->b.getValue().compareTo(a.getValue())).forEach(e->
                        LOGGER.info(e.getKey()+"\t"+e.getValue())
        );
    }
}
