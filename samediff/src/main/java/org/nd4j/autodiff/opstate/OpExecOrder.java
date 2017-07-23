package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 5/1/17.
 */
@Data
@Builder
public class OpExecOrder {
    private List<OpExecAction> actions;


    /**
     *
     * @return
     */
    public List<OpState> opStates() {
        List<OpState> ret = new ArrayList<>(actions.size());
        for(OpExecAction opExecAction : actions) {
            ret.add(opExecAction.getOpState());
        }

        return ret;
    }

    public List<String> opNames() {
        List<String> ret = new ArrayList<>(actions.size());
        for(OpExecAction opExecOrder : actions) {
            ret.add(opExecOrder.getOpState().getOpName());
        }

        return ret;
    }





}
