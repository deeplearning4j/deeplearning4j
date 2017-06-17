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

package org.apdplat.word.solr;

import java.io.BufferedReader;
import java.io.Reader;
import java.io.StringReader;
import java.util.Map;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.util.TokenizerFactory;
import org.apache.lucene.util.AttributeFactory;
import org.apdplat.word.lucene.ChineseWordTokenizer;
import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lucene中文分词器工厂
 * @author 杨尚川
 */
public class ChineseWordTokenizerFactory extends TokenizerFactory {
    private static final Logger LOGGER = LoggerFactory.getLogger(ChineseWordTokenizerFactory.class);
    private Segmentation segmentation = null;
    public ChineseWordTokenizerFactory(Map<String, String> args){
        super(args);
        if(args != null){
            String conf = args.get("conf");
            if(conf != null){
                //强制覆盖默认配置
                WordConfTools.forceOverride(conf);
            }else{
                LOGGER.info("没有指定conf参数");
            }
            String algorithm = args.get("segAlgorithm");
            if(algorithm != null){
                try{
                    SegmentationAlgorithm segmentationAlgorithm = SegmentationAlgorithm.valueOf(algorithm);
                    segmentation = SegmentationFactory.getSegmentation(segmentationAlgorithm);
                    LOGGER.info("使用指定分词算法："+algorithm);
                }catch(Exception e){
                    LOGGER.error("参数segAlgorithm指定的值错误："+algorithm);
                    LOGGER.error("参数segAlgorithm可指定的值有：");
                    for(SegmentationAlgorithm sa : SegmentationAlgorithm.values()){
                        LOGGER.error("\t"+sa.name());
                    }
                }
            }else{
                LOGGER.info("没有指定segAlgorithm参数");
            }
        }
        if(segmentation == null){
            segmentation = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMaximumMatching);
            LOGGER.info("使用默认分词算法："+SegmentationAlgorithm.BidirectionalMaximumMatching);
        }
    }
    @Override
    public Tokenizer create(AttributeFactory af) {
        return new ChineseWordTokenizer(segmentation);
    }
}