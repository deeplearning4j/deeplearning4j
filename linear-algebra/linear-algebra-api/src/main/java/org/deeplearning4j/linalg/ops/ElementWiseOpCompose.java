package org.deeplearning4j.linalg.ops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Aggregation of element wise operations
 *
 * @author Adam Gibson
 */
public class ElementWiseOpCompose {
    private List<ElementWiseOp> opsToExecute = new ArrayList<>();
    private boolean reverse = false;

    public ElementWiseOpCompose op(ElementWiseOp op) {
        opsToExecute.add(op);
        return this;
    }


    public void exec() {
        if(reverse)
            Collections.reverse(opsToExecute);


        for(ElementWiseOp op : opsToExecute)
            op.exec();

    }




}
