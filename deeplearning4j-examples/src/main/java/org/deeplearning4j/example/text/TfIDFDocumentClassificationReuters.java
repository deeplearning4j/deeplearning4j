package org.deeplearning4j.example.text;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.ReutersNewsGroupsDataSetIterator;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

/**
 * @author Adam Gibson
 */
public class TfIDFDocumentClassificationReuters {

    private static Logger log = LoggerFactory.getLogger(TfIDFDocumentClassificationReuters.class);


    public static void main(String[] args) throws Exception {
          DataSetIterator iter = new ReutersNewsGroupsDataSetIterator(10,10,true);
          while(iter.hasNext()) {
              DataSet next = iter.next();
              log.info("Data dims " + next.numInputs() + " labels " + next.numOutcomes());
          }
    }


}
