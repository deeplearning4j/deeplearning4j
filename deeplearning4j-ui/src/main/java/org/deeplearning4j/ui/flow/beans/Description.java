package org.deeplearning4j.ui.flow.beans;

import lombok.Data;

import java.io.Serializable;

/**
 * Description bean holds few lines worth text description for any layer + optional actionId link, linked to specific REST request.
 *
 * @author raver119@gmail.com
 */
@Data
public class Description implements Serializable {
    private String mainLine;
    private String subLine;

    /*
        Action ID is used via REST request to /flow/action/{id}
        call to that should open separate interface instance, and provide additional info about objects
     */
    private long actionId;
}
