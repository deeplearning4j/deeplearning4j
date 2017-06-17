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

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * 相似度排名
 * @author 杨尚川
 */
public interface SimilarityRanker extends Similarity{
    /**
     * 计算源文本和目标文本的相似度
     * 根据相似度分值对目标文本进行排序
     * @param source 源文本
     * @param targets 目标文本
     * @return 相似度排名结果列表
     */
    default Hits rank(String source, List<String> targets){
        return rank(source, targets, Integer.MAX_VALUE);
    }

    /**
     * 计算源文本和目标文本的相似度
     * 根据相似度分值对目标文本进行排序
     * 获取排名结果最高的topN项
     * @param source 源文本
     * @param targets 目标文本
     * @param topN 相似度排名结果列表只保留相似度分值最高的topN项
     * @return 相似度排名结果列表
     */
    default Hits rank(String source, List<String> targets, int topN){
        Hits hits = new Hits(topN>targets.size()?targets.size():topN);
        targets
            .parallelStream()
            .map(target -> {
                double score = similarScore(source, target);
                Hit hit = new Hit();
                hit.setText(target);
                hit.setScore(score);
                return hit;
            })
            .sorted()
            .limit(topN)
            .collect(Collectors.toList())
            .forEach(hit -> hits.addHit(hit));
        return hits;
    }

    static void main(String[] args) throws Exception{
        TextSimilarity textSimilarity = new EditDistanceTextSimilarity();
        List<String> sentences = Files.readAllLines(Paths.get("src/test/resources/dic.txt"))
                .stream()
                .map(line -> line.trim())
                .filter(line -> line.length() > 1)
                .collect(Collectors.toList());
        System.out.println("开始计算 "+sentences.size()+" 句话的文本相似度");
        AtomicInteger i = new AtomicInteger();
        sentences.forEach(sentence -> {
            AtomicInteger j = new AtomicInteger();
            System.out.println("****************************************************************************************************");
            System.out.println(i.incrementAndGet() + "、文本 "+sentence+" 的相似度文本：");
            textSimilarity.rank(sentence, sentences, 15).getHits().forEach(hit -> System.out.println("\t" + j.incrementAndGet() + "、" + hit.getScore() + " " + hit.getText()));
            System.out.println("****************************************************************************************************");
        });
    }
}
