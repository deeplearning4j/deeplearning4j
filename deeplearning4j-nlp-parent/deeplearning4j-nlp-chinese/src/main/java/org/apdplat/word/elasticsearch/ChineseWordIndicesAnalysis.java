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

package org.apdplat.word.elasticsearch;

import org.apache.lucene.analysis.Tokenizer;
import org.apdplat.word.lucene.ChineseWordAnalyzer;
import org.apdplat.word.lucene.ChineseWordTokenizer;
import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.elasticsearch.common.component.AbstractComponent;
import org.elasticsearch.common.inject.Inject;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.analysis.AnalyzerScope;
import org.elasticsearch.index.analysis.PreBuiltAnalyzerProviderFactory;
import org.elasticsearch.index.analysis.PreBuiltTokenizerFactoryFactory;
import org.elasticsearch.index.analysis.TokenizerFactory;
import org.elasticsearch.indices.analysis.IndicesAnalysisService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * 中文分词索引分析组件
 * @author 杨尚川
 */
public class ChineseWordIndicesAnalysis extends AbstractComponent {
    private static final Logger LOGGER = LoggerFactory.getLogger(ChineseWordIndicesAnalysis.class);
    private Segmentation analyzerSegmentation = null;
    private Segmentation tokenizerSegmentation = null;
    @Inject
    public ChineseWordIndicesAnalysis(Settings settings, IndicesAnalysisService indicesAnalysisService) {
        super(settings);
        Object index = settings.getAsStructuredMap().get("index");
        if(index != null && index instanceof Map){
            Map indexMap = (Map)index;
            Object analysis = indexMap.get("analysis");
            if(analysis != null && analysis instanceof Map){
                Map analysisMap = (Map)analysis;
                Object analyzer = analysisMap.get("analyzer");
                Object tokenizer = analysisMap.get("tokenizer");
                if(analyzer != null && analyzer instanceof Map){
                    Map analyzerMap = (Map)analyzer;
                    Object _default = analyzerMap.get("default");
                    if(_default != null && _default instanceof Map){
                        Map _defaultMap = (Map)_default;
                        Object type = _defaultMap.get("type");
                        Object segAlgorithm = _defaultMap.get("segAlgorithm");
                        if(segAlgorithm != null && type != null && "word".equals(type.toString())){
                            LOGGER.info("analyzer使用指定分词算法："+segAlgorithm.toString());
                            analyzerSegmentation = SegmentationFactory.getSegmentation(SegmentationAlgorithm.valueOf(segAlgorithm.toString()));
                        }
                    }
                }
                if(tokenizer != null && tokenizer instanceof Map){
                    Map tokenizerMap = (Map)tokenizer;
                    Object _default = tokenizerMap.get("default");
                    if(_default != null && _default instanceof Map){
                        Map _defaultMap = (Map)_default;
                        Object type = _defaultMap.get("type");
                        Object segAlgorithm = _defaultMap.get("segAlgorithm");
                        if(segAlgorithm != null && type != null && "word".equals(type.toString())){
                            LOGGER.info("tokenizer使用指定分词算法："+segAlgorithm.toString());
                            tokenizerSegmentation = SegmentationFactory.getSegmentation(SegmentationAlgorithm.valueOf(segAlgorithm.toString()));
                        }
                    }
                }
            }            
        }
        if(analyzerSegmentation == null){
            LOGGER.info("没有为word analyzer指定segAlgorithm参数");
            LOGGER.info("analyzer使用默认分词算法："+SegmentationAlgorithm.BidirectionalMaximumMatching);
            analyzerSegmentation = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMaximumMatching);
        }
        if(tokenizerSegmentation == null){
            LOGGER.info("没有为word tokenizer指定segAlgorithm参数");
            LOGGER.info("tokenizer使用默认分词算法："+SegmentationAlgorithm.BidirectionalMaximumMatching);
            tokenizerSegmentation = SegmentationFactory.getSegmentation(SegmentationAlgorithm.BidirectionalMaximumMatching);
        }
        // 注册分析器
        indicesAnalysisService.analyzerProviderFactories()
                .put("word", new PreBuiltAnalyzerProviderFactory("word", AnalyzerScope.GLOBAL, 
                        new ChineseWordAnalyzer(analyzerSegmentation)));
        // 注册分词器
        indicesAnalysisService.tokenizerFactories()
                .put("word", new PreBuiltTokenizerFactoryFactory(new TokenizerFactory() {
            @Override
            public String name() {
                return "word";
            }
            @Override
            public Tokenizer create() {
                return new ChineseWordTokenizer(tokenizerSegmentation);
            }
        }));        
    }
}