package org.ansj.app.keyword;

public class Keyword implements Comparable<Keyword> {
    private String name;
    private double score;
    private double idf;
    private int freq;

    public Keyword(String name, int docFreq, double weight) {
        this.name = name;
        this.idf = Math.log(1 + 10000.0 / (docFreq + 1));
        this.score = idf * weight;
        freq++;
    }

    public Keyword(String name, double score) {
        this.name = name;
        this.score = score;
        this.idf = score;
        freq++;
    }

    public void updateWeight(int weight) {
        this.score += weight * idf;
        freq++;
    }

    public int getFreq() {
        return freq;
    }

    @Override
    public int compareTo(Keyword o) {
        if (this.score < o.score) {
            return 1;
        } else {
            return -1;
        }

    }

    @Override
    public boolean equals(Object obj) {

        if (obj instanceof Keyword) {
            Keyword k = (Keyword) obj;
            return k.name.equals(name);
        } else {
            return false;
        }
    }

    @Override
    public String toString() {
        return name + "/" + score;// "="+score+":"+freq+":"+idf;
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

    public void setScore(double score) {
        this.score = score;
    }

}
