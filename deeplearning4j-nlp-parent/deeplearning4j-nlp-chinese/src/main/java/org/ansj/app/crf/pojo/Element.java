package org.ansj.app.crf.pojo;

import org.ansj.app.crf.Config;
import org.ansj.app.crf.Model;

public class Element {

    public char name;
    private int tag = -1;
    public int len = 1;
    public String nature;

    public float[] tagScore;

    public int[] from;

    public Element(char name) {
        this.name = name;
    }

    public Element(Character name, int tag) {
        this.name = name;
        this.tag = tag;
    }

    public int getTag() {
        return tag;
    }

    public Element updateTag(int tag) {
        this.tag = tag;
        return this;
    }

    public Element updateNature(String nature) {
        this.nature = nature;
        return this;
    }

    @Override
    public String toString() {
        return name + "/" + len + "/" + tag;
    }

    public char getName() {
        return name;
    }

    /**
     * 获得可见的名称
     * 
     * @return
     */
    public String nameStr() {
        if (name >= 130 && name < 140) {
            return ("num" + (name - 130));
        } else if (name >= 140 && name < 150) {
            return ("en" + (name - 140));
        } else {
            return String.valueOf(name);
        }
    }

    public void maxFrom(Model model, Element element) {
        if (from == null) {
            from = new int[Config.TAG_NUM];
        }
        float[] pTagScore = element.tagScore;
        for (int i = 0; i < Config.TAG_NUM; i++) {
            float maxValue = 0;
            for (int j = 0; j < Config.TAG_NUM; j++) {

                float value = (pTagScore[j] + tagScore[i]) + model.tagRate(j, i);

                if (tagScore.length > Config.TAG_NUM) {
                    value += tagScore[Config.TAG_NUM + j * Config.TAG_NUM + i];
                }

                if (value > maxValue) {
                    maxValue = value;
                    from[i] = j;
                }

            }

            tagScore[i] = maxValue;
        }
    }

}
