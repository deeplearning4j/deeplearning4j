package org.ansj.domain;

import java.io.Serializable;

/**
 * 新词发现,实体名
 * 
 * @author ansj
 * 
 */
public class NewWord implements Serializable {
    /**
     * 
     */
    private static final long serialVersionUID = 7226797287286838356L;
    // 名字
    private String name;
    // 分数
    private double score;
    // 词性
    private Nature nature;
    // 总词频
    private int allFreq;
    // 此词是否被激活
    private boolean isActive;

    public NewWord(String name, Nature nature, double score) {
        this.name = name;
        this.nature = nature;
        this.score = score;
        this.allFreq = 1;
    }

    public NewWord(String name, Nature nature) {
        this.name = name;
        this.nature = nature;
        this.allFreq = 1;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public double getScore() {
        return score;
    }

    public Nature getNature() {
        return nature;
    }

    public void setNature(Nature nature) {
        this.nature = nature;
    }

    /**
     * 更新发现权重,并且更新词性
     * 
     * @param version
     * @param i
     * @param tn
     */
    public void update(Nature nature, int freq) {
        this.score += score * freq;
        this.allFreq += freq;
        if (Nature.NW != nature) {
            this.nature = nature;
        }
    }

    @Override
    public String toString() {
        return this.name + "\t" + this.score + "\t" + this.getNature().natureStr;
    }

    public int getAllFreq() {
        return allFreq;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public boolean isActive() {
        return isActive;
    }

    public void setActive(boolean isActive) {
        this.isActive = isActive;
    }

}
