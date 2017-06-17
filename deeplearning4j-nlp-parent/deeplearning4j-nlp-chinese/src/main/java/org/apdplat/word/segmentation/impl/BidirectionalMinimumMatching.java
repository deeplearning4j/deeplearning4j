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

import java.util.List;
import java.util.Map;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.segmentation.Word;

/**
 * 基于词典的双向最小匹配算法
 * Dictionary-based bidirectional minimum matching algorithm
 * @author 杨尚川
 */
public class BidirectionalMinimumMatching extends AbstractSegmentation{
    private static final AbstractSegmentation MIM = (AbstractSegmentation)SegmentationFactory.getSegmentation(SegmentationAlgorithm.MinimumMatching);
    private static final AbstractSegmentation RMIM = (AbstractSegmentation)SegmentationFactory.getSegmentation(SegmentationAlgorithm.ReverseMinimumMatching);

    @Override
    public SegmentationAlgorithm getSegmentationAlgorithm() {
        return SegmentationAlgorithm.BidirectionalMinimumMatching;
    }
    @Override
    public List<Word> segImpl(final String text) {
        //逆向最小匹配
        List<Word> wordsRMIM = RMIM.seg(text);
        if(!ngramEnabled()){
            return wordsRMIM;
        }
        //正向最小匹配
        List<Word> wordsMIM = MIM.seg(text);
        //如果分词结果都一样，则直接返回结果
        if(wordsRMIM.size() == wordsMIM.size() 
                && wordsRMIM.equals(wordsMIM)){            
            return wordsRMIM;
        }
        
        //如果分词结果不一样，则利用ngram消歧
        Map<List<Word>, Float> words = ngram(wordsRMIM, wordsMIM);        
      
        //如果分值都一样，则选择逆向最小匹配
        float score = words.get(wordsRMIM);
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("逆向最小匹配：" + wordsRMIM.toString() + " : ngram分值=" + score);
        }
        //最终结果
        List<Word> result = wordsRMIM;
        //最小分值
        float max = score;
        
        score = words.get(wordsMIM);
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("正向最小匹配：" + wordsMIM.toString() + " : ngram分值=" + score);
        }
        //只有正向最小匹配的分值大于逆向最小匹配，才会被选择
        if(score > max){
            result = wordsMIM;
            max = score;
        }

        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("最大分值：" + max + ", 消歧结果：" + result);
        }
        return result;
    }
    public static void main(String[] args){
        String text = "古人对于写文章有个基本要求，叫做“有物有序”。“有物”就是要有内容，“有序”就是要有条理。";
        if(args !=null && args.length == 1){
            text = args[0];
        }
        BidirectionalMinimumMatching m = new BidirectionalMinimumMatching();
        System.out.println(m.seg(text).toString());
    }
}
