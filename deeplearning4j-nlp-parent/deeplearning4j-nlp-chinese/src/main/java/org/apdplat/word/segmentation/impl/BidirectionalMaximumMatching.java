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
 * 基于词典的双向最大匹配算法
 * Dictionary-based bidirectional maximum matching algorithm
 * @author 杨尚川
 */
public class BidirectionalMaximumMatching extends AbstractSegmentation{
    private static final AbstractSegmentation MM = (AbstractSegmentation)SegmentationFactory.getSegmentation(SegmentationAlgorithm.MaximumMatching);
    private static final AbstractSegmentation RMM = (AbstractSegmentation)SegmentationFactory.getSegmentation(SegmentationAlgorithm.ReverseMaximumMatching);

    @Override
    public SegmentationAlgorithm getSegmentationAlgorithm() {
        return SegmentationAlgorithm.BidirectionalMaximumMatching;
    }
    @Override
    public List<Word> segImpl(final String text) {
        //逆向最大匹配
        List<Word> wordsRMM = RMM.seg(text);
        if(!ngramEnabled()){
            return wordsRMM;
        }
        //正向最大匹配
        List<Word> wordsMM = MM.seg(text);
        //如果分词结果都一样，则直接返回结果
        if(wordsRMM.size() == wordsMM.size() 
                && wordsRMM.equals(wordsMM)){            
            return wordsRMM;
        }
        
        //如果分词结果不一样，则利用ngram消歧
        Map<List<Word>, Float> words = ngram(wordsRMM, wordsMM);        
      
        //如果分值都一样，则选择逆向最大匹配
        float score = words.get(wordsRMM);
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("逆向最大匹配：" + wordsRMM.toString() + " : ngram分值=" + score);
        }
        //最终结果
        List<Word> result = wordsRMM;
        //最大分值
        float max = score;
        
        score = words.get(wordsMM);
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("正向最大匹配：" + wordsMM.toString() + " : ngram分值=" + score);
        }
        //只有正向最大匹配的分值大于逆向最大匹配，才会被选择
        if(score > max){
            result = wordsMM;
            max = score;
        }

        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("最大分值：" + max + ", 消歧结果：" + result);
        }
        return result;
    }
    public static void main(String[] args){
        String text = "APDPlat的雏形可以追溯到2008年，并于4年后即2012年4月9日在GITHUB开源 。APDPlat在演化的过程中，经受住了众多项目的考验，一直追求简洁优雅，一直对架构、设计和代码进行重构优化。 ";
        if(args !=null && args.length == 1){
            text = args[0];
        }
        BidirectionalMaximumMatching m = new BidirectionalMaximumMatching();
        System.out.println(m.seg(text).toString());
    }
}
