package org.deeplearning4j.scaleout.perform.text;

import org.deeplearning4j.scaleout.perform.BaseWorkPerformerFactory;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;

/**
 *
 * Word count work performer
 * @author Adam Gibson
 */
public class WordCountWorkPerformerFactory extends BaseWorkPerformerFactory {

    @Override
    public WorkerPerformer instantiate() {
        return new WordCountWorkPerformer();
    }
}
