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

package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.apache.commons.math3.distribution.*;
import org.deeplearning4j.arbiter.optimize.distribution.LogUniformDistribution;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

/**
 * Custom JSON serializer for Apache commons RealDistribution instances.
 * The custom serializer is set up to use the built-in c
 */
public class RealDistributionSerializer extends JsonSerializer<RealDistribution> {

    @Override
    public void serialize(RealDistribution d, JsonGenerator j, SerializerProvider serializerProvider)
                    throws IOException {
        Class<?> c = d.getClass();
        String s = c.getSimpleName();

        j.writeStartObject();
        j.writeStringField("distribution", s);


        if (c == BetaDistribution.class) {
            BetaDistribution bd = (BetaDistribution) d;
            j.writeNumberField("alpha", bd.getAlpha());
            j.writeNumberField("beta", bd.getBeta());
        } else if (c == CauchyDistribution.class) {
            CauchyDistribution cd = (CauchyDistribution) d;
            j.writeNumberField("median", cd.getMedian());
            j.writeNumberField("scale", cd.getScale());
        } else if (c == ChiSquaredDistribution.class) {
            ChiSquaredDistribution cd = (ChiSquaredDistribution) d;
            j.writeNumberField("dof", cd.getDegreesOfFreedom());
        } else if (c == ExponentialDistribution.class) {
            ExponentialDistribution ed = (ExponentialDistribution) d;
            j.writeNumberField("mean", ed.getMean());
        } else if (c == FDistribution.class) {
            FDistribution fd = (FDistribution) d;
            j.writeNumberField("numeratorDof", fd.getNumeratorDegreesOfFreedom());
            j.writeNumberField("denominatorDof", fd.getDenominatorDegreesOfFreedom());
        } else if (c == GammaDistribution.class) {
            GammaDistribution gd = (GammaDistribution) d;
            j.writeNumberField("shape", gd.getShape());
            j.writeNumberField("scale", gd.getScale());
        } else if (c == LevyDistribution.class) {
            LevyDistribution ld = (LevyDistribution) d;
            j.writeNumberField("mu", ld.getLocation());
            j.writeNumberField("c", ld.getScale());
        } else if (c == LogNormalDistribution.class) {
            LogNormalDistribution ln = (LogNormalDistribution) d;
            j.writeNumberField("scale", ln.getScale());
            j.writeNumberField("shape", ln.getShape());
        } else if (c == NormalDistribution.class) {
            NormalDistribution nd = (NormalDistribution) d;
            j.writeNumberField("mean", nd.getMean());
            j.writeNumberField("stdev", nd.getStandardDeviation());
        } else if (c == ParetoDistribution.class) {
            ParetoDistribution pd = (ParetoDistribution) d;
            j.writeNumberField("scale", pd.getScale());
            j.writeNumberField("shape", pd.getShape());
        } else if (c == TDistribution.class) {
            TDistribution td = (TDistribution) d;
            j.writeNumberField("dof", td.getDegreesOfFreedom());
        } else if (c == TriangularDistribution.class) {
            TriangularDistribution td = (TriangularDistribution) d;
            j.writeNumberField("a", td.getSupportLowerBound());
            j.writeNumberField("b", td.getMode());
            j.writeNumberField("c", td.getSupportUpperBound());
        } else if (c == UniformRealDistribution.class) {
            UniformRealDistribution u = (UniformRealDistribution) d;
            j.writeNumberField("lower", u.getSupportLowerBound());
            j.writeNumberField("upper", u.getSupportUpperBound());
        } else if (c == WeibullDistribution.class) {
            WeibullDistribution wb = (WeibullDistribution) d;
            j.writeNumberField("alpha", wb.getShape());
            j.writeNumberField("beta", wb.getScale());
        } else if (c == LogUniformDistribution.class){
            LogUniformDistribution lud = (LogUniformDistribution) d;
            j.writeNumberField("min", lud.getMin());
            j.writeNumberField("max", lud.getMax());
        } else {
            throw new UnsupportedOperationException("Unknown or not supported RealDistribution: " + d.getClass());
        }

        j.writeEndObject();
    }
}
