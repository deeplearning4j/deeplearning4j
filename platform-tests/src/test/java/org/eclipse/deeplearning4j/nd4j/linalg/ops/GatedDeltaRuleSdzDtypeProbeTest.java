/*
 * ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.api.buffer.DataType;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Metadata-only probe of a staged SDZ: prints the dtypes of every variable feeding the
 * {@code gated_delta_rule} ops so a dtype-mismatch failure can be diagnosed WITHOUT
 * executing the model (no GPU, no native plan). Run with
 * {@code -Dgdr.probe.sdz=/path/to/model.sdz}; skipped otherwise so large staged
 * artifacts never become ordinary test fixtures.
 */
public class GatedDeltaRuleSdzDtypeProbeTest {

    @Test
    public void printGatedDeltaRuleInputDtypes() throws Exception {
        String configured = System.getProperty("gdr.probe.sdz");
        assumeTrue(configured != null && !configured.isBlank(),
                "Set -Dgdr.probe.sdz to the staged model.sdz to probe");
        System.out.println("GDR_PROBE loading (metadata only, no weights spill): " + configured);
        SameDiff graph = SDZSerializer.load(new java.io.File(configured), true);
        try {
            int printed = 0;
            for (DifferentialFunction op : graph.ops()) {
                if (!"gated_delta_rule".equals(op.opName())) {
                    continue;
                }
                String[] inputNames = graph.getInputsForOp(op);
                if (inputNames != null) {
                    for (String inputName : inputNames) {
                        SDVariable in = graph.getVariable(inputName);
                        DataType dt = in.dataType();
                        System.out.printf("GDR_PROBE op=%s input=%s dtype=%s shape=%s%n",
                                op.getOwnName(), inputName, dt, java.util.Arrays.toString(in.getShape()));
                    }
                }
                printed++;
            }
            System.out.println("GDR_PROBE gated_delta_rule op instances: " + printed);
            assumeTrue(printed > 0, "Staged SDZ contains no gated_delta_rule ops");
        } finally {
            graph.close();
        }
    }
}
