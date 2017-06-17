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

package org.apdplat.word.lucene;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Queue;
import java.util.concurrent.LinkedTransferQueue;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.recognition.StopWord;
import org.apdplat.word.segmentation.Word;
import org.apdplat.word.tagging.AntonymTagging;
import org.apdplat.word.tagging.PinyinTagging;
import org.apdplat.word.tagging.SynonymTagging;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lucene中文分词器
 * @author 杨尚川
 */
public class ChineseWordTokenizer extends Tokenizer {
    private static final Logger LOGGER = LoggerFactory.getLogger(ChineseWordTokenizer.class);
    
    private final CharTermAttribute charTermAttribute = addAttribute(CharTermAttribute.class);
    private final OffsetAttribute offsetAttribute = addAttribute(OffsetAttribute.class);
    private final PositionIncrementAttribute positionIncrementAttribute = addAttribute(PositionIncrementAttribute.class);

    private static final boolean FULL_PINYIN = WordConfTools.getBoolean("tagging.pinyin.full", false);
    private static final boolean ACRONYM_PINYIN = WordConfTools.getBoolean("tagging.pinyin.acronym", false);
    private static final boolean SYNONYM = WordConfTools.getBoolean("tagging.synonym", false);
    private static final boolean ANTONYM = WordConfTools.getBoolean("tagging.antonym", false);

    private Segmentation segmentation = null;
    private BufferedReader reader = null;
    private final Queue<Word> words = new LinkedTransferQueue<>();
    private final Queue<String> tokens = new LinkedTransferQueue<>();
    private int startOffset=0;
        
    public ChineseWordTokenizer() {
        segmentation = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMinimumMatching);
    }

    public ChineseWordTokenizer(String segmentationAlgorithm) {
        try{
            SegmentationAlgorithm sa = SegmentationAlgorithm.valueOf(segmentationAlgorithm);
            this.segmentation = SegmentationFactory.getSegmentation(sa);
        }catch(Exception e){
            this.segmentation = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMinimumMatching);
        }
    }

    public ChineseWordTokenizer(SegmentationAlgorithm segmentationAlgorithm) {
        this.segmentation = SegmentationFactory.getSegmentation(segmentationAlgorithm);
    }

    public ChineseWordTokenizer(Segmentation segmentation) {
        this.segmentation = segmentation;
    }

    private Word getWord() throws IOException {
        Word word = words.poll();
        if(word == null){
            if(reader==null){
                reader = new BufferedReader(input);
            }
            String line;
            while( (line = reader.readLine()) != null ){
                words.addAll(segmentation.seg(line));
            }
            startOffset = 0;
            word = words.poll();
        }
        return word;
    }

    private String getToken() throws IOException {
        String token = tokens.poll();
        if(token == null){
            Word word = getWord();
            if (word != null) {
                int positionIncrement = 1;
                //忽略停用词
                while (StopWord.is(word.getText())) {
                    positionIncrement++;
                    startOffset += word.getText().length();
                    if(LOGGER.isDebugEnabled()) {
                        LOGGER.debug("忽略停用词：" + word.getText());
                    }
                    word = getWord();
                    if (word == null) {
                        return null;
                    }
                }
                offsetAttribute.setOffset(startOffset, startOffset + word.getText().length());
                positionIncrementAttribute.setPositionIncrement(positionIncrement);
                startOffset += word.getText().length();
                tokens.offer(word.getText());
                //拼音标注
                if (FULL_PINYIN || ACRONYM_PINYIN) {
                    PinyinTagging.process(Arrays.asList(word));
                    if(FULL_PINYIN && !"".equals(word.getFullPinYin())) {
                        tokens.offer(word.getFullPinYin());
                    }
                    if(ACRONYM_PINYIN  && !"".equals(word.getAcronymPinYin())) {
                        tokens.offer(word.getAcronymPinYin());
                    }
                }
                //同义标注
                if (SYNONYM) {
                    SynonymTagging.process(Arrays.asList(word));
                    StringBuilder synonym = new StringBuilder();
                    word.getSynonym().forEach(w -> {
                        if (!"".equals(w.getText())) {
                            tokens.offer(w.getText());
                        }
                        synonym.append(w.getText()).append(" ");
                    });
                }
                //反义标注
                if (ANTONYM) {
                    AntonymTagging.process(Arrays.asList(word));
                    StringBuilder antonym = new StringBuilder();
                    word.getAntonym().forEach(w -> {
                        if (!"".equals(w.getText())) {
                            tokens.offer(w.getText());
                        }
                        antonym.append(w.getText()).append(" ");
                    });
                }
                token = tokens.poll();
            }
        }
        return token;
    }

    @Override
    public final boolean incrementToken() throws IOException {
        String token = getToken();
        if (token != null) {
            charTermAttribute.setEmpty().append(token);
            return true;
        }
        return false;
    }
}