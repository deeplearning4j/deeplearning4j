package org.nd4j.linalg.api.ops;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract base class for {@link Module}
 * that handles Dynamic ops and handles nesting.
 *
 * This is a logical unit for defining layers
 * very similar to pytorch's modules, or tensorflow's layers.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public abstract class BaseModule extends DynamicCustomOp implements Module {
    private List<Module> modules = new ArrayList<>();

    public BaseModule(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, List<Module> modules) {
        super(opName, inputs, outputs, tArguments, iArguments);
        this.modules = modules;
    }

    public BaseModule(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, List<Module> modules) {
        super(opName, sameDiff, args, inPlace);
        this.modules = modules;
    }

    @Override
    public Module[] subModules() {
        return modules.toArray(new Module[modules.size()]);
    }

    @Override
    public void addModule(Module module) {
        modules.add(module);
    }




}
