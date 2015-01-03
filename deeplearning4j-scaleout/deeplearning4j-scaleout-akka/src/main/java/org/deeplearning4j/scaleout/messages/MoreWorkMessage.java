package org.deeplearning4j.scaleout.messages;

import java.io.Serializable;

/**
 * Asks for more work from the batch actor
 */
public class MoreWorkMessage implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = -4149080551476702735L;
    private static MoreWorkMessage INSTANCE = new MoreWorkMessage();

    private MoreWorkMessage() {
		super();
	}

    public static MoreWorkMessage getInstance() {
        return INSTANCE;
    }



	

}
