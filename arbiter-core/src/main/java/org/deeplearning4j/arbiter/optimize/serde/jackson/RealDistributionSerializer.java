package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.apache.commons.math3.distribution.*;
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
    public void serialize(RealDistribution realDistribution, JsonGenerator j, SerializerProvider serializerProvider) throws IOException {
        Class<?> c = realDistribution.getClass();
        String s = c.getSimpleName();

        j.writeStartObject();
        j.writeStringField("distribution", s);


        if(c == BetaDistribution.class){
            BetaDistribution bd = (BetaDistribution)realDistribution;
            j.writeNumberField("alpha", bd.getAlpha());
            j.writeNumberField("beta", bd.getBeta());
        } else if( c == CauchyDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == ChiSquaredDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == ExponentialDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == FDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == GammaDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == LevyDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == LogNormalDistribution.class ){
            LogNormalDistribution ln = (LogNormalDistribution)realDistribution;
            j.writeNumberField("scale",ln.getScale());
            j.writeNumberField("shape", ln.getShape());
        } else if( c == NormalDistribution.class ){
            NormalDistribution nd = (NormalDistribution)realDistribution;
            j.writeNumberField("mean", nd.getMean());
            j.writeNumberField("stdev", nd.getStandardDeviation());
        } else if( c == ParetoDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == TDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == TriangularDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == UniformRealDistribution.class ){
            UniformRealDistribution u = (UniformRealDistribution)realDistribution;
            j.writeNumberField("lower", u.getSupportLowerBound());
            j.writeNumberField("upper", u.getSupportUpperBound());
        } else if( c == WeibullDistribution.class ) {
            throw new UnsupportedOperationException("Not yet implemented");
        } else {
            throw new UnsupportedOperationException("Unknown or not supported RealDistribution: " + realDistribution.getClass());
        }

        j.writeEndObject();
    }
}
