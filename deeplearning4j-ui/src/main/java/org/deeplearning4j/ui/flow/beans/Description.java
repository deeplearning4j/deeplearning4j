package org.deeplearning4j.ui.flow.beans;

import lombok.Data;

import java.io.Serializable;

/**
 * Description bean holds few lines worth text description for any layer
 *
 * @author raver119@gmail.com
 */
@Data
public class Description implements Serializable {
    private final static long serialVersionUID = 119L;
    private String mainLine = "";
    private String subLine = "";
}
