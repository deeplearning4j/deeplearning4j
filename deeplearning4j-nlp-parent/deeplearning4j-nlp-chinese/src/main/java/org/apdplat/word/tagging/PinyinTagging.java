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

import net.sourceforge.pinyin4j.PinyinHelper;
import net.sourceforge.pinyin4j.format.HanyuPinyinCaseType;
import net.sourceforge.pinyin4j.format.HanyuPinyinOutputFormat;
import net.sourceforge.pinyin4j.format.HanyuPinyinToneType;
import net.sourceforge.pinyin4j.format.HanyuPinyinVCharType;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.segmentation.Word;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * 拼音标注
 * @author 杨尚川
 */
public class PinyinTagging {
    private PinyinTagging(){}

    private static final Logger LOGGER = LoggerFactory.getLogger(PinyinTagging.class);

    public static void process(List<Word> words){
        for (Word word : words){
            String wordText = word.getText();
            word.setFullPinYin(getFullPinYin(wordText));
            word.setAcronymPinYin(getAcronymPinYin(wordText));
        }
    }

    private static String getAcronymPinYin(String words){
        return getPinYin(words, true);
    }

    private static String getFullPinYin(String words) {
        return getPinYin(words, false);
    }

    private static String getPinYin(String words, boolean acronym) {
        HanyuPinyinOutputFormat format = new HanyuPinyinOutputFormat();
        format.setCaseType(HanyuPinyinCaseType.LOWERCASE);
        format.setToneType(HanyuPinyinToneType.WITHOUT_TONE);
        format.setVCharType(HanyuPinyinVCharType.WITH_U_UNICODE);
        char[] chars = words.trim().toCharArray();
        StringBuilder result = new StringBuilder();
        try {
            for (char c : chars) {
                if (Character.toString(c).matches("[\u4e00-\u9fa5]+")) {
                    String[] pinyinStringArray = PinyinHelper.toHanyuPinyinStringArray(c, format);
                    if(acronym){
                        result.append(pinyinStringArray[0].charAt(0));
                    }else {
                        result.append(pinyinStringArray[0]);
                    }
                }else{
                    return null;
                }
            }
        } catch (Exception e) {
            LOGGER.error("拼音标注失败", e);
        }
        return result.toString();
    }

    public static void main(String[] args) {
        List<Word> words = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMaximumMatching).seg("《速度与激情7》的中国内地票房自4月12日上映以来，在短短两周内突破20亿人民币");
        System.out.println(words);
        PinyinTagging.process(words);
        System.out.println(words);
    }
}
