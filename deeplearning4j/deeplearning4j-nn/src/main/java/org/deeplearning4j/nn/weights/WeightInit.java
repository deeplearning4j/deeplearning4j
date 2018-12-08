/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.weights;

import com.google.common.base.CaseFormat;
import com.sun.xml.internal.bind.v2.TODO;
import org.deeplearning4j.nn.conf.distribution.Distribution;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Weight initialization scheme
 * <p>
 * <b>DISTRIBUTION</b>: Sample weights from a provided distribution<br>
 * <p>
 * <b>ZERO</b>: Generate weights as zeros<br>
 * <p>
 * <b>ONES</b>: All weights are set to 1
 * <p>
 * <b>SIGMOID_UNIFORM</b>: A version of XAVIER_UNIFORM for sigmoid activation functions. U(-r,r) with r=4*sqrt(6/(fanIn + fanOut))
 * <p>
 * <b>NORMAL</b>: Normal/Gaussian distribution, with mean 0 and standard deviation 1/sqrt(fanIn).
 * This is the initialization recommented in Klambauer et al. 2017, "Self-Normalizing Neural Network". Equivalent to
 * DL4J's XAVIER_FAN_IN and LECUN_NORMAL (i.e. Keras' "lecun_normal")
 * <p>
 * <b>LECUN_UNIFORM</b> Uniform U[-a,a] with a=3/sqrt(fanIn).
 * <p>
 * <b>UNIFORM</b>: Uniform U[-a,a] with a=1/sqrt(fanIn). "Commonly used heuristic" as per Glorot and Bengio 2010
 * <p>
 * <b>XAVIER</b>: As per Glorot and Bengio 2010: Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)
 * <p>
 * <b>XAVIER_UNIFORM</b>: As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
 * <p>
 * <b>XAVIER_FAN_IN</b>: Similar to Xavier, but 1/fanIn -> Caffe originally used this.
 * <p>
 * <b>XAVIER_LEGACY</b>: Xavier weight init in DL4J up to 0.6.0. XAVIER should be preferred.
 * <p>
 * <b>RELU</b>: He et al. (2015), "Delving Deep into Rectifiers". Normal distribution with variance 2.0/nIn
 * <p>
 * <b>RELU_UNIFORM</b>: He et al. (2015), "Delving Deep into Rectifiers". Uniform distribution U(-s,s) with s = sqrt(6/fanIn)
 * <p>
 * <b>IDENTITY</b>: Weights are set to an identity matrix. Note: can only be used with square weight matrices
 * <p>
 * <b>VAR_SCALING_NORMAL_FAN_IN</b> Gaussian distribution with mean 0, variance 1.0/(fanIn)
 * <p>
 * <b>VAR_SCALING_NORMAL_FAN_OUT</b> Gaussian distribution with mean 0, variance 1.0/(fanOut)
 * <p>
 * <b>VAR_SCALING_NORMAL_FAN_AVG</b> Gaussian distribution with mean 0, variance 1.0/((fanIn + fanOut)/2)
 * <p>
 * <b>VAR_SCALING_UNIFORM_FAN_IN</b> Uniform U[-a,a] with a=3.0/(fanIn)
 * <p>
 * <b>VAR_SCALING_UNIFORM_FAN_OUT</b> Uniform U[-a,a] with a=3.0/(fanOut)
 * <p>
 * <b>VAR_SCALING_UNIFORM_FAN_AVG</b> Uniform U[-a,a] with a=3.0/((fanIn + fanOut)/2)
 * <p>
 *
 * @author Adam Gibson
 */
public enum WeightInit {
    DISTRIBUTION, ZERO, ONES, SIGMOID_UNIFORM, NORMAL, LECUN_NORMAL, UNIFORM, XAVIER, XAVIER_UNIFORM, XAVIER_FAN_IN, XAVIER_LEGACY, RELU,
    RELU_UNIFORM, IDENTITY, LECUN_UNIFORM, VAR_SCALING_NORMAL_FAN_IN, VAR_SCALING_NORMAL_FAN_OUT, VAR_SCALING_NORMAL_FAN_AVG,
    VAR_SCALING_UNIFORM_FAN_IN, VAR_SCALING_UNIFORM_FAN_OUT, VAR_SCALING_UNIFORM_FAN_AVG;

    /**
     * Create an instance of the weight initialization function
     *
     * @param distribution Distribution of the weights (Only used in case DISTRIBUTION)
     * @return a new {@link IWeightInit} instance
     */
    public IWeightInit getWeightInitFunction(Distribution distribution) {
        switch (this) {
            case DISTRIBUTION:
                return new WeightInitConstant();
            case ZERO:
                return new WeightInitConstant(0.0);
            case ONES:
                return new WeightInitConstant(1.0);
            case SIGMOID_UNIFORM:
                return new WeightInitConstant();
            case NORMAL:
                return new WeightInitConstant();
            case LECUN_NORMAL:
                return new WeightInitConstant();
            case UNIFORM:
                return new WeightInitConstant();
            case XAVIER:
                return new WeightInitConstant();
            case XAVIER_UNIFORM:
                return new WeightInitConstant();
            case XAVIER_FAN_IN:
                return new WeightInitConstant();
            case XAVIER_LEGACY:
                return new WeightInitConstant();
            case RELU:
                return new WeightInitConstant();
            case RELU_UNIFORM:
                return new WeightInitConstant();
            case IDENTITY:
                return new WeightInitConstant();
            case LECUN_UNIFORM:
                return new WeightInitConstant();
            case VAR_SCALING_NORMAL_FAN_IN:
                return new WeightInitConstant();
            case VAR_SCALING_NORMAL_FAN_OUT:
                return new WeightInitConstant();
            case VAR_SCALING_NORMAL_FAN_AVG:
                return new WeightInitConstant();
            case VAR_SCALING_UNIFORM_FAN_IN:
                return new WeightInitConstant();
            case VAR_SCALING_UNIFORM_FAN_OUT:
                return new WeightInitConstant();
            case VAR_SCALING_UNIFORM_FAN_AVG:
                return new WeightInitConstant();

            default:
                throw new UnsupportedOperationException("Unknown or not supported weight initialization function: " + this);
        }
    }

    public static void main(String[] args) {
        for (WeightInit wi : WeightInit.values()) {

            final String docStr = "/\n" +
                    "  Weight initialization scheme\n" +
                    "  \n" +
                    "  DISTRIBUTION: Sample weights from a provided distribution<br>\n" +
                    "  \n" +
                    "  ZERO: Generate weights as zeros<br>\n" +
                    "  \n" +
                    "  ONES: All weights are set to 1\n" +
                    "  \n" +
                    "  SIGMOID_UNIFORM: A version of XAVIER_UNIFORM for sigmoid activation functions. U(-r,r) with r=4sqrt(6/(fanIn + fanOut))\n" +
                    "  \n" +
                    "  NORMAL: Normal/Gaussian distribution, with mean 0 and standard deviation 1/sqrt(fanIn).\n" +
                    "  This is the initialization recommented in Klambauer et al. 2017, \"Self-Normalizing Neural Network\". Equivalent to\n" +
                    "  DL4J's XAVIER_FAN_IN and LECUN_NORMAL (i.e. Keras' \"lecun_normal\")\n" +
                    "  \n" +
                    "  LECUN_UNIFORM Uniform U[-a,a] with a=3/sqrt(fanIn).\n" +
                    "  \n" +
                    "  UNIFORM: Uniform U[-a,a] with a=1/sqrt(fanIn). \"Commonly used heuristic\" as per Glorot and Bengio 2010\n" +
                    "  \n" +
                    "  XAVIER: As per Glorot and Bengio 2010: Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)\n" +
                    "  \n" +
                    "  XAVIER_UNIFORM: As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))\n" +
                    "  \n" +
                    "  XAVIER_FAN_IN: Similar to Xavier, but 1/fanIn -> Caffe originally used this.\n" +
                    "  \n" +
                    "  XAVIER_LEGACY: Xavier weight init in DL4J up to 0.6.0. XAVIER should be preferred.\n" +
                    "  \n" +
                    "  RELU: He et al. (2015), \"Delving Deep into Rectifiers\". Normal distribution with variance 2.0/nIn\n" +
                    "  \n" +
                    "  RELU_UNIFORM: He et al. (2015), \"Delving Deep into Rectifiers\". Uniform distribution U(-s,s) with s = sqrt(6/fanIn)\n" +
                    "  \n" +
                    "  IDENTITY: Weights are set to an identity matrix. Note: can only be used with square weight matrices\n" +
                    "  \n" +
                    "  VAR_SCALING_NORMAL_FAN_IN Gaussian distribution with mean 0, variance 1.0/(fanIn)\n" +
                    "  \n" +
                    "  VAR_SCALING_NORMAL_FAN_OUT Gaussian distribution with mean 0, variance 1.0/(fanOut)\n" +
                    "  \n" +
                    "  VAR_SCALING_NORMAL_FAN_AVG Gaussian distribution with mean 0, variance 1.0/((fanIn + fanOut)/2)\n" +
                    "  \n" +
                    "  VAR_SCALING_UNIFORM_FAN_IN Uniform U[-a,a] with a=3.0/(fanIn)\n" +
                    "  \n" +
                    "  VAR_SCALING_UNIFORM_FAN_OUT Uniform U[-a,a] with a=3.0/(fanOut)\n" +
                    "  \n" +
                    "  VAR_SCALING_UNIFORM_FAN_AVG Uniform U[-a,a] with a=3.0/((fanIn + fanOut)/2)\n" +
                    "  \n" +
                    " \n" +
                    "  @author Adam Gibson";
            String classDoc = "TODO";
            final Pattern pattern = Pattern.compile("(\\s*" + wi.name() + ")(.*)");
            for (String str : docStr.split("\n")) {
                // System.out.println("str: " + str.trim());
                final Matcher m = pattern.matcher(str);
                if (m.matches()) {
                    // System.out.println("Matched " + wi + " vs " + m.group(2));
                    classDoc = m.group(2);
                    break;
                }
            }

            final String className = "WeightInit" + CaseFormat.UPPER_UNDERSCORE.to(CaseFormat.UPPER_CAMEL, wi.name());
            final String classDef = "package org.deeplearning4j.nn.weights;\n" +
                    "\n" +
                    "import org.nd4j.linalg.api.ndarray.INDArray;\n" +
                    "\n" +
                    "/**\n" +
                    " * " + classDoc.trim() + "\n" +
                    " *\n" +
                    " * @author Adam Gibson\n" +
                    " */\n" +
                    "public class " + className + " implements IWeightInit {\n" +
                    "\n" +
                    "\n" +
                    "\n" +
                    "    @Override\n" +
                    "    public void init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {\n" +
                    "        paramView.assign(value);\n" +
                    "    }\n" +
                    "}";
            System.out.println(classDef);

            final String filename = "E:\\Software projects\\java\\deeplearning4j\\deeplearning4j\\deeplearning4j-nn\\src\\main\\java\\org\\deeplearning4j\\nn\\weights\\" + className + ".java";
            try (PrintWriter out = new PrintWriter(filename)) {
                out.println(classDef);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }
}
