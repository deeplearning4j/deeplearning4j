package org.nd4j.graph.intermediate;

import lombok.*;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.graph.OpClass;
import org.nd4j.linalg.primitives.ImmutablePair;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.List;

/**
 * This class is intermediate representation of operation
 *
 * @author raver119@gmail.com
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TOp {
    @Builder.Default private List<TIndex> inputs = new ArrayList<>();

    // we can use the same TIndex here, but we don't really need it. Only input nodes should care about indices
    @Builder.Default private List<Integer> outputs = new ArrayList<>();

    // may be set externally, i.e. in TF graph
    @Builder.Default private String name = "";

    // exists only after mapping
    private int id;

    // opName in libnd4j op space
    private String opName;

    // opNum is applicable only to legacy XYZ ops
    private int opNum;

    // op group basically
    private OpClass opClass;

    // this value defines, if this Node belongs to some scope
    @Builder.Default private boolean scoped = false;


    @Builder.Default private int scopeId = 0;
    @Builder.Default private String scopeName = "";

    // parameters for opd
    private OpState opState;

    public TOp(int id) {
        this.id = id;
    }

    public void addInput(@NonNull TIndex input) {
        inputs.add(input);
    }

    public void addInput(int input) {
        inputs.add(TIndex.makeOf(input));
    }

    public void addInput(int input, int index) {
        inputs.add(TIndex.makeOf(input, index));
    }

    @Override
    public String toString() {
        return "TOp{" +
                "inputs=" + inputs +
                ", opName='" + name + '\'' +
                ", id=" + id +
                ", opName='" + opName + '\'' +
                '}';
    }
}
