package org.ansj.app.crf;

import org.ansj.app.crf.pojo.Element;
import org.ansj.util.MatrixUtil;
import org.nlpcn.commons.lang.util.StringUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 分词
 * 
 * @author ansj
 * 
 */
public class SplitWord {

    private Model model = null;

    public SplitWord(Model model) {
        this.model = model;
    };

    public List<String> cut(char[] chars) {
        return cut(new String(chars));
    }

    public List<String> cut(String line) {

        if (StringUtil.isBlank(line)) {
            return Collections.emptyList();
        }

        List<Element> elements = vterbi(line);

        List<String> result = new ArrayList<>();

        Element e = null;
        int begin = 0;
        int end = 0;
        int size = elements.size() - 1;
        for (int i = 0; i < elements.size(); i++) {
            e = elements.get(i);
            switch (e.getTag()) {
                case 0:
                    end += e.len;
                    result.add(line.substring(begin, end));
                    begin = end;
                    break;
                case 1:
                    end += e.len;
                    while (i < size && (e = elements.get(++i)).getTag() != 3) {
                        end += e.len;
                    }
                    end += e.len;
                    result.add(line.substring(begin, end));
                    begin = end;
                default:
                    break;
            }
        }
        return result;
    }

    private List<Element> vterbi(String line) {
        List<Element> elements = Config.wordAlert(line);

        int length = elements.size();

        if (length == 0) { // 避免空list，下面get(0)操作越界
            return elements;
        }
        if (length == 1) {
            elements.get(0).updateTag(0);
            return elements;
        }

        /**
         * 填充图
         */
        for (int i = 0; i < length; i++) {
            computeTagScore(elements, i);
        }

        // 如果是开始不可能从 m，e开始 ，所以将它设为一个很小的值
        elements.get(0).tagScore[2] = -1000;
        elements.get(0).tagScore[3] = -1000;

        for (int i = 1; i < length; i++) {
            elements.get(i).maxFrom(model, elements.get(i - 1));
        }

        // 末位置只能从S,E开始
        // 末位置只能从0,3开始

        Element next = elements.get(elements.size() - 1);

        Element self = null;

        int maxStatus = next.tagScore[0] > next.tagScore[3] ? 0 : 3;

        next.updateTag(maxStatus);

        maxStatus = next.from[maxStatus];

        // 逆序寻找
        for (int i = elements.size() - 2; i > 0; i--) {
            self = elements.get(i);
            self.updateTag(maxStatus);
            maxStatus = self.from[self.getTag()];
            next = self;
        }
        elements.get(0).updateTag(maxStatus);

        // printElements(elements) ;

        return elements;

    }

    private void computeTagScore(List<Element> elements, int index) {

        char[][] feautres = model.getConfig().makeFeatureArr(elements, index);

        //TODO: set 20 很大吧!
        float[] tagScore = new float[20]; //Config.TAG_NUM*Config.TAG_NUM+Config.TAG_NUM

        for (int i = 0; i < feautres.length; i++) {
            MatrixUtil.dot(tagScore, model.getFeature(feautres[i]));
        }

        elements.get(index).tagScore = tagScore;
    }

    /**
     * 随便给一个词。计算这个词的内聚分值，可以理解为计算这个词的可信度
     * 
     * @param word
     */
    public float cohesion(String word) {

        if (word.length() == 0) {
            return Integer.MIN_VALUE;
        }

        List<Element> elements = Config.wordAlert(word);

        for (int i = 0; i < elements.size(); i++) {
            computeTagScore(elements, i);
        }

        float value = elements.get(0).tagScore[1];

        int len = elements.size() - 1;

        for (int i = 1; i < len; i++) {
            value += elements.get(i).tagScore[2];
        }

        value += elements.get(len).tagScore[3];

        if (value < 0) {
            return 1;
        } else {
            value += 1;
        }

        return value;
    }

}
