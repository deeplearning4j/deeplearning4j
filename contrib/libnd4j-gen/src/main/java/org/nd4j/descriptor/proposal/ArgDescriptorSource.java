package org.nd4j.descriptor.proposal;

import org.nd4j.ir.OpNamespace;

import java.util.List;
import java.util.Map;

public interface ArgDescriptorSource {

    Map<String, List<ArgDescriptorProposal>> getProposals();

    OpNamespace.OpDescriptor.OpDeclarationType typeFor(String name);


}
