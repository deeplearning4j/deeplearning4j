/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.glove.actor;

import java.io.Serializable;

/**
 * Created by agibsonccc on 12/7/14.
 */
public class SentenceWork implements Serializable {
    private int id;
    private String sentence;

    public SentenceWork(int id, String sentence) {
        this.id = id;
        this.sentence = sentence;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getSentence() {
        return sentence;
    }

    public void setSentence(String sentence) {
        this.sentence = sentence;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SentenceWork)) return false;

        SentenceWork that = (SentenceWork) o;

        if (id != that.id) return false;
        return !(sentence != null ? !sentence.equals(that.sentence) : that.sentence != null);

    }

    @Override
    public int hashCode() {
        int result = id;
        result = 31 * result + (sentence != null ? sentence.hashCode() : 0);
        return result;
    }
}
