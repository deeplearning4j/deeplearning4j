package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Alex Black
 */
@Slf4j
public abstract class BaseOptimizerSet implements OptimizerSet {


    @Override
    public List<Optimizer> getOptimizers() {
        Method[] methods = this.getClass().getDeclaredMethods();
        List<Optimizer> out = new ArrayList<>(methods.length);
        for(Method m : methods){
            int modifiers = m.getModifiers();
            Class<?> retType = m.getReturnType();
            if(retType != null && Modifier.isPublic(modifiers) && Optimizer.class.isAssignableFrom(retType) ){
                try {
                    Optimizer o = (Optimizer) m.invoke(null);
                    out.add(o);
                } catch (IllegalAccessException | InvocationTargetException e) {
                    log.warn("Could not create optimizer from method: {}", m, e);
                }
            }
        }

        Class<?>[] declaredClasses = this.getClass().getDeclaredClasses();
        for(Class<?> c : declaredClasses){
            int modifiers = c.getModifiers();
            if(Modifier.isPublic(modifiers) && !Modifier.isAbstract(modifiers) && Optimizer.class.isAssignableFrom(c)){
                try{
                    out.add((Optimizer) c.newInstance());
                } catch (IllegalAccessException | InstantiationException e) {
                    log.warn("Could not create optimizer from inner class: {}", c, e);
                }
            }
        }

        return out;
    }
}
