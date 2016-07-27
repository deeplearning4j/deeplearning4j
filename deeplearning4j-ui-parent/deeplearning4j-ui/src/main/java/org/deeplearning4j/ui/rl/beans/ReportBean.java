package org.deeplearning4j.ui.rl.beans;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ReportBean implements Serializable {
    private final static long serialVersionUID = 119L;
    private long epochId;
    private double reward;
}
