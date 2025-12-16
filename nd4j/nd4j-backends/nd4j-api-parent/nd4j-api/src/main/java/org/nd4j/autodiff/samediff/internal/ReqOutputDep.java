package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@AllArgsConstructor
@Builder
class ReqOutputDep extends Dep {
    protected String outputName;
}
