package org.nd4j.descriptor.proposal;

import lombok.Builder;
import lombok.Data;
import org.nd4j.ir.OpNamespace;

@Builder
@Data
public class ArgDescriptorProposal {

    private OpNamespace.ArgDescriptor descriptor;

    private String sourceLine;

    private double proposalWeight;

    private String sourceOfProposal;
}
