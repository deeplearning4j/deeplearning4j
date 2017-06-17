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

import org.apdplat.word.WordSegmenter;
import org.apdplat.word.recognition.RecognitionTool;
import org.apdplat.word.segmentation.PartOfSpeech;
import org.apdplat.word.segmentation.Word;
import org.apdplat.word.util.AutoDetector;
import org.apdplat.word.util.GenericTrie;
import org.apdplat.word.util.ResourceLoader;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * 词性标注
 * @author 杨尚川
 */
public class PartOfSpeechTagging {
    private PartOfSpeechTagging(){}

    private static final Logger LOGGER = LoggerFactory.getLogger(PartOfSpeechTagging.class);
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
                LOGGER.info("初始化词性标注器");
                int count = 0;
                for (String line : lines) {
                    try {
                        String[] attr = line.split(":");
                        GENERIC_TRIE.put(attr[0], attr[1]);
                        count++;
                    } catch (Exception e) {
                        LOGGER.error("错误的词性数据：" + line);
                    }
                }
                LOGGER.info("词性标注器初始化完毕，词性数据条数：" + count);
            }

            @Override
            public void add(String line) {
                try {
                    String[] attr = line.split(":");
                    GENERIC_TRIE.put(attr[0], attr[1]);
                } catch (Exception e) {
                    LOGGER.error("错误的词性数据：" + line);
                }
            }

            @Override
            public void remove(String line) {
                try {
                    String[] attr = line.split(":");
                    GENERIC_TRIE.remove(attr[0]);
                } catch (Exception e) {
                    LOGGER.error("错误的词性数据：" + line);
                }
            }

        }, WordConfTools.get("part.of.speech.dic.path", "classpath:part_of_speech_dic.txt"));
    }
    public static void process(List<Word> words){
        words.parallelStream().forEach(word->{
            if(word.getPartOfSpeech()!=null){
                if(LOGGER.isDebugEnabled()) {
                    LOGGER.debug("忽略已经标注过的词：{}", word);
                }
                return;
            }
            String wordText = word.getText();
            String pos = GENERIC_TRIE.get(wordText);
            if(pos == null){
                //识别英文
                if(RecognitionTool.isEnglish(wordText)){
                    pos = "w";
                }
                //识别数字
                if(RecognitionTool.isNumber(wordText)){
                    pos = "m";
                }
                //中文数字
                if(RecognitionTool.isChineseNumber(wordText)){
                    pos = "mh";
                }
                //识别小数和分数
                if(RecognitionTool.isFraction(wordText)){
                    if(wordText.contains(".")||wordText.contains("．")||wordText.contains("·")){
                        pos = "mx";
                    }
                    if(wordText.contains("/")||wordText.contains("／")){
                        pos = "mf";
                    }
                }
                //识别数量词
                if(RecognitionTool.isQuantifier(wordText)){
                    //分数
                    if(wordText.contains("‰")||wordText.contains("%")||wordText.contains("％")){
                        pos = "mf";
                    }
                    //时间量词
                    else if(wordText.contains("时")||wordText.contains("分")||wordText.contains("秒")){
                        pos = "tq";
                    }
                    //日期量词
                    else if(wordText.contains("年")||wordText.contains("月")||wordText.contains("日")
                            ||wordText.contains("天")||wordText.contains("号")){
                        pos = "tdq";
                    }
                    //数量词
                    else{
                        pos = "mq";
                    }
                }
            }
            word.setPartOfSpeech(PartOfSpeech.valueOf(pos));
        });
    }

    public static void main(String[] args) {
        List<Word> words = WordSegmenter.segWithStopWords("我爱中国，我爱杨尚川");
        System.out.println("未标注词性："+words);
        //词性标注
        PartOfSpeechTagging.process(words);
        System.out.println("标注词性："+words);
    }
}
