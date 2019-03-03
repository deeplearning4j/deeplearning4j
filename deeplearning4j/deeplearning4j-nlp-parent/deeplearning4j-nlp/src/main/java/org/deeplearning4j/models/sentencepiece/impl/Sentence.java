package org.deeplearning4j.models.sentencepiece.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@AllArgsConstructor
@NoArgsConstructor
@Data
public class Sentence {
    private String string;
    private long second;
}
