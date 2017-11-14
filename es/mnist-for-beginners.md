---
title: MNIST para principiantes
layout: default
---
<!-- Begin Inspectlet Embed Code -->
<script type="text/javascript" id="inspectletjs">
window.__insp = window.__insp || [];
__insp.push(['wid', 1755897264]);
(function() {
function ldinsp(){if(typeof window.__inspld != "undefined") return; window.__inspld = 1; var insp = document.createElement('script'); insp.type = 'text/javascript'; insp.async = true; insp.id = "inspsync"; insp.src = ('https:' == document.location.protocol ? 'https' : 'http') + '://cdn.inspectlet.com/inspectlet.js'; var x = document.getElementsByTagName('script')[0]; x.parentNode.insertBefore(insp, x); };
setTimeout(ldinsp, 500); document.readyState != "complete" ? (window.attachEvent ? window.attachEvent('onload', ldinsp) : window.addEventListener('load', ldinsp, false)) : ldinsp();
})();
</script>
<!-- End Inspectlet Embed Code -->

# MNIST para principiantes

En este tutorial, vamos a clasificar el conjunto de datos (dataset) de MNIST, el "Hola Mundo" del machine learning.

Tabla de contenidos

1. [Introduci&oacute;n](#introduci&oacute;n)
2. [El dataset de MNIST](#el-dataset-de-mnist)
3. [Configurando el ejemplo de MNIST](#configurando-el-ejemplo-de-mnist)
4. [Construyendo nuestra red neuronal](#construyendo-nuestra-red-neuronal)
5. [Entrenando el modelo](#entrenando-el-modelo)
6. [Evaluando los resultados](#evaluando-los-resultados)
7. [Conclusi&oacute;n](#conclusi&oacute;n)

El tiempo estimado para completarlo es de 30 minutos.

## Introduci&oacute;n

![rederizado MNIST](../img/mnist_render.png)

MNIST es una base de datos que contiene im&aacute;genes de n&uacute;meros escritos a mano, donde cada imagen est&aacute; etiquetada con un n&uacute;mero entero. Es usada para medir el rendimiento de algoritmos de machine learning. Deep Learning rinde bastante bien con MNIST, llegando a m&aacute;s de un 99,7% de precisi&oacute;n.

Vamos a utilizar MNIST para entrenar una red neuronal que inspeccione cada imagen y prediga el d&iacute;gito. El primer paso es instalar Deeplearning4j.

<a href="quickstart" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', 'quickstart', 'click');">EMPEZANDO CON DEEPLEARNING4J</a>

## El dataset de MNIST

El dataset de MNIST contiene un **conjunto de entrenamiento** de 60.000 im&aacute;genes y un **conjunto de pruebas** de 10.000 ejemplos. El conjunto de entrenamiento es utilizado para ense&ntilde;ar al algoritmo a predecir la etiqueta correcta, el n&uacute;mero entero, mientras que el conjunto de pruebas es usado para comprobar c&oacute;mo de precisas puede hacer la red entrenada sus estimaciones.

En el mundo del machine learning, esto es llamado [entrenamiento supervisado](https://en.wikipedia.org/wiki/Supervised_learning), porque tenemos la respuesta correcta para las im&aacute;genes que estamos tratando de adivinar. El conjunto de entrenamiento puede actuar por tanto como supervisor o profesor, corrigiendo a la red neuronal cuando sus predicciones son err&oacute;neas. 

## Configurando el ejemplo de MNIST

Hemos empaquetado el tutorial de MNIST en Maven, as&iacute; que no hay necesidad de escribir c&oacute;digo. Por favor, abra el IntelliJ para comenzar. (Para descargar IntelliJ, vea nuestra [gu&iacute;a r&aacute;pida](./quickstart))

Abra la carpeta etiquetada como `dl4j-examples`. Vaya a los directorios <kbd>src</kbd> → <kbd>main</kbd> → <kbd>java</kbd> → <kbd>feedforward</kbd> → <kbd>mnist</kbd>, y abra el fichero `MLPMnistSingleLayerExample.java`.

![ejemplo de configuracion mlp capa unica](../img/mlp_mnist_single_layer_example_setup.png)

En este fichero, configuraremos una red neuronal, entrenaremos un modelo y evaluaremos los resultados. Ser&aacute; de utilida ver este c&oacute;digo al mismo tiempo que el tutorial.

### Estableciendo las variables

``` java
    final int numRows = 28; // Numero de filas de la matriz.
    final int numColumns = 28; // Numero de columnas de la matriz.
    int outputNum = 10; // Numero de posibles salidas (e.g. etiquetas de 0 a 9).
    int batchSize = 128; // Cuantos ejemplos utilizar en cada paso.
    int rngSeed = 123; // Este generador de numeros aleatorios utiliza una semilla para asegurar que los mismos pesos son utilizados durante el entrenamiento. Explicaremos el porqu&eacute; de esto mas adelante.
    int numEpochs = 15; // Una epoca es una pasada completa a través de un dataset.
```

En nuestro ejemplo, cada imagen MNIST es de 28x28 p&iacute;xeles, lo que significa que nuestros datos de entrada es una matriz de 28 **numRows** por 28 **numColumns** (las matrices son la estructura de datos fundamental del deep learning). Adem&aacute;s, MNIST contiene 10 posibles salidas (las etiquetas numeradas 0 - 9) que es nuestro **outputNum**.

El **batchSize** y **numEpochs** tienen que elegirse basado en la experiencia; aprendes qu&eacute; funciona mediante la experimentaci&oacute;n. Un tama&ntilde;o de bloque (batchSize) grande hace un entrenamiento r&aacute;pido, mientras que m&aacute;s &eacute;pocas (numEpochs), o pasadas a trav&eacute;s del dataset, resulta en una mejor precisi&oacute;n.

Sin embargo, hay un resultado degradante una vez pasado un cierto n&uacute;mero de &eacute;pocas, por lo que hay un compromiso entre precisi&oacute;n y velocidad de entrenamiento. En general, tendr&aacute; que experimentar para descubrir los valores &oacute;ptimos. Hemos puesto unos valores por defecto razonables en este ejemplo.

### Cargando los datos de MNIST

``` java
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
```

La clase llamada `DataSetIterator` se usa para cargar el dataset de MNIST. Creamos un dataset `mnistTrain` para **entrenar el modelo** y otro dataset `mnistTest` para **evaluar la precisi&oacute;n** de nuestro modelo despu&eacute;s del entrenamiento. El modelo, dicho sea de paso, se refiere a los par&aacute;metros de la red neuronal. Estos par&aacute;metros son coeficientes que procesan la se&ntilde;al de entrada, y son ajustados según la red aprende hasta que pueden predecir la etiqueta correcta para cada imagen; en ese punto, tiene un modelo preciso.

## Construyendo nuestra red neuronal

Construiremos una red neuronal retroalimentada basada en el [art&iacute;culo de Xavier Glorot y Yoshua Bengio](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf). Para nuestro prop&oacute;sito, vamos a empezar con un ejemplo b&aacute;sico con s&oacute;lo una capa oculta. Sin embargo, como regla r&aacute;pida, cuanto m&aacute;s profunda sea la red (es decir: cuantas m&aacute;s capas), m&aacute;s complejidad y matices puede capturar para producir unos resultados precisos.

![red una capa oculta](../img/onelayer.png)

Guarde esta imagen en su cabeza porque es lo que estamos construyendo, una red neuronal de una sola capa.

### Estableciendo los hiperpar&aacute;metros

Para cualquier red neuronal que construya con Deeplearning4j, la base es la [clase NeuralNetConfiguration](../neuralnet-configuration). Ah&iacute; es donde configura los hiperpar&aacute;metros las cantidades que definen la arquitectura y c&oacute;mo el algoritmo aprende. Intuitivamente, cada hiperpar&aacute;metro es como un ingrediente de una comida, una comida que puede salir muy bien, o muy mal... Afortunadamente, podrá ajustar los hiperparametros si no producen los resultados correctos.

``` java
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
```
 
##### .seed(rngSeed)
Este par&aacute;metro usa una inicializaci&oacute;n de pesos generada aleatoriamente espec&iacute;fica. Si ejecuta un ejemplo m&uacute;ltiples veces, y genera unos nuevos pesos aleatorios cada vez que empieza, entonces el resultado de su red - precisi&oacute;n y Valor-F - pueden variar mucho, porque diferentes pesos iniciales pueden conducir al algoritmo a diferentes m&iacute;nimos locales en el espacio de error (errorscape). Manteniendo los mismos pesos aleatorios le permite aislar los efectos de ajustar otros hiperpar&aacute;metros m&aacute;s claramente, mientras otras condiciones se mantienen igual.

##### .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
El descenso por el gradiente estoc&aacute;stico (SGD en ingl&eacute;s) es un m&eacute;todo com&uacute;n para optimizar la funci&oacute;n de coste. Para aprender m&aacute;s sobre el SGD y otros algoritmos de optimizaci&oacute;n que ayudan a minimizar el error, le recomendamos [el curso de machine learning de Andrew Ng](https://www.coursera.org/learn/machine-learning) y la definici&oacute;n de SGD en nuestro [glosario](../glossary#stochastic-gradient-descent)

##### .iterations(1)
Cada iteraci&oacute;n, para una red neuronal, es un paso del entrenamiento; es decir, una actualizaci&oacute;n de los pesos del modelo. La red es expuesta a los datos, hace predicciones sobre los datos, y entonces corrige sus propios par&aacute;metros bas&aacute;ndose en cu&aacute;nto de erroneas fueron sus predicciones. As&iacute; que m&aacute;s iteraciones permiten a la red hacer m&aacute;s pasos y aprender m&aacute;s, minimizando el error.

##### .learningRate(0.006)
Esta l&iacute;nea establece la tasa de aprendizaje, que es el tama&ntilde;o de los ajustes hechos en los pesos con cada iteraci&oacute;n, el tama&ntilde;o de cada paso. Una tasa de aprendizaje alta hace a una red atravesar el espacio de error (erroscape) r&aacute;pido, pero tambi&eacute;n la hace susceptible de pasarse de largo el punto de m&iacute;nimo error. Una tasa de aprendizaje baja es m&aacute;s probable que encuentre el m&iacute;nimo, pero lo har&aacute; lentamente, porque va haciendo peque&ntilde;os ajustes en los pesos.

##### .updater(Updater.NESTEROVS).momentum(0.9)
Momentum es un factor adicional en determinar cu&aacute;nto de r&aacute;pido converger&aacute; el algoritmo de optimizaci&oacute;n al punto &oacute;ptimo. Momentum afecta a la direcci&oacute;n que los pesos son ajustados, as&iacute; que en el c&oacute;digo lo consideramos un actualizador de pesos `updater`.

##### .regularization(true).l2(1e-4)
La regularizaci&oacute;n es una t&eacute;cnica para prevenir lo que se denomina sobreajuste (overfitting en ingl&eacute;s). El sobreajuste es cuando el modelo se ajusta muy bien a los datos de entrenamiento, pero desempe&ntilde;a mal en datos reales tan pronto como es expuesto a datos que no ha visto antes.

Usamos la regularizaci&oacute;n L2, que previene que pesos individuales puedan tener demasiada influencia en los resultados totales.

##### .list()
La funci&oacute;n list especifica el n&uacute;mero de capas que tendr&aacute; la red; esta funci&oacute;n replica la configuraci&oacute;n n veces y construye una configuraci&oacute;n en forma de capas.

De nuevo, si algo de lo de arriba fue confuso, le recomendamos [el curso de machine learning de Andrew Ng](https://www.coursera.org/learn/machine-learning).

### Construyendo capas
No vamos a profundizar en la investigaci&oacute;n de cada hiperpar&aacute;metro (p. ej: activation, weightInit); S&oacute;lamente vamos a intentar dar una breve descripci&oacute;n de lo que hacen. Sin embargo, si&eacute;ntase libre de leer el [art&iacute;culo de Xavier Glorot y Yoshua Bengio](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) para aprender el porqu&eacute; de estas materias.

![red una capa oculta etiquetada](../img/onelayer_labeled.png)

``` java
    .layer(0, new DenseLayer.Builder()
            .nIn(numRows * numColumns) // Number of input datapoints.
            .nOut(1000) // Number of output datapoints.
            .activation("relu") // Activation function.
            .weightInit(WeightInit.XAVIER) // Weight initialization.
            .build())
    .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(1000)
            .nOut(outputNum)
            .activation("softmax")
            .weightInit(WeightInit.XAVIER)
            .build())
    .pretrain(false).backprop(true)
    .build();
```

Entonces &iquest;qu&eacute; es exactamente una capa oculta?

Cada nodo (los c&iacute;rculos) en la capa oculta representan una caracter&iacute;stica de un d&iacute;gito escrito a mano en el dataset de MNIST. Por ejemplo, imagine que est&aacute; mirando el n&uacute;mero 6. Un nodo puede representar los bordes redondeados, otro nodo puede representar la intersecci&oacute;n de las lineas curvas, etc&eacute;tera. Estas caracter&iacute;sticas est&aacute;n ponderadas por importancia mediante los coeficientes del modelo, y recombinados en cada capa oculta para ayudar a predecir si el n&uacute;mero escrito a mano es de hecho un 6. Cuantas m&aacute;s capas de nodos tenga, m&aacute;s complejidad y matices capturar&aacute;n para hacer mejores predicciones.

Puede pensar en una capa como "oculta" porque, mientras que puede ver los datos de entrada introduci&eacute;ndose en la red y la decisi&oacute;n saliendo, es dif&iacute;cil para los humanos descifrar c&oacute;mo y por qu&eacute; una red neuronal procesa los datos en su interior. Los par&aacute;metros de un modelo de red neuronal son simples y largos vectores de números, legibles para las m&aacute;quinas.

## Entrenando el modelo

Ahora que el modelo está construido, empezemos con el entrenamiento. En la parte de arriba a la derecha de IntelliJ, pulse en la flecha verde. Esto ejecutar&aacute; el c&oacute;digo descrito arriba.

![ejemplo entrenamiento red una capa](../img/mlp_mnist_single_layer_example_training.png)

Puede tomar varios minutos hasta que el entrenamiento se complete, dependiendo de su hardware.

## Evaluando los resultados

![resultado ejemplo entrenamiento red una capa](../img/mlp_mnist_single_layer_example_results.png)

**Exactitud** - El porcentaje de im&aacute;genes de MNIST que fueron correctamente identificadas por el modelo.  
**Precisi&oacute;n** - El n&uacute;mero de verdaderos positivos dividido por el n&uacute;mero de falsos positivos y verdaderos positivos.  
**Sensibilidad** - El n&uacute;mero de verdaderos positivos dividido por el n&uacute;mero de verdaderos positivos y el n&uacute;mero de falsos negativos.  
**Valor-F** - Media ponderada de la precisi&oacute;n y la sensibilidad.

La **exactidud** mide el modelo general.

**Precisión, sensibilidad y Valor-F** miden la **relevancia** del modelo. Por ejemplo, ser&iacute;a peligoroso que un c&aacute;ncer no va a reaparecer (es decir, un falso negativo) porque la persona podr&iacute;a no buscar m&aacute;s tratamiento. Por esto, ser&iacute;a inteligente elegir un modelo que evitase los falsos positivos (es decir, mayor precisi&oacute;n, sensibilidad y Valor-F) incluso si la **exactitud** general es menor.

## Conclusi&oacute;n

&iexcl;Y aqu&iacute; lo tiene! En este punto, ha entrenado satisfactoriamente una red neuronal sin ning&uacute;n conocimiento en el campo de la visi&oacute;n por computador con un 97,1% de exactitud. El rendimiento del estado del arte es incluso mejor que eso, y puede mejorar el modelo ajustando a&uacute;n m&aacute;s los hiperpar&aacute;metros.

Comparado con otros frameworks, Deeplearning4j sobresale en lo siguiente:
* Integraci&oacute;n con los frameworks m&aacute;s importantes de la JVM como Spark, Hadoop y Kafka a gran escala.
* Optimizado para ejecutarse en CPUs y/o GPUs distribuidas.
* Atendiendo a las comunidades Java y Scala.
* Soporte comercial para despliegues empresariales.

Si tiene alguna otra cuesti&oacute;n, por favor &uacute;nase a nuestro [chat de soporte en Gitter](https://gitter.im/deeplearning4j/deeplearning4j).

  <ul class="categorized-view view-col-3">
    <li>
      <h5>OTROS TUTORIALES DE DEEPLEARNING4J</h5>
      <a href="../neuralnet-overview">Introducci&oacute;n a las Redes Neuronales</a>
      <a href="../restrictedboltzmannmachine">M&aacute;quinas de Boltzmann restringidas</a>
      <a href="../eigenvector">Eigenvectors, Covarianza, PCA and Entrop&iacute;a</a>
      <a href="../lstm">LSTMs and Redes Recurrentes</a>
      <a href="../linear-regression">Redes Neuronales y Regresi&oacute;n</a>
      <a href="../convolutionalnets">Redes Convolucionales</a>
    </li>

    <li>
      <h5>RECURSOS RECOMENDADOS</h5>
      <a href="https://www.coursera.org/learn/machine-learning/home/week/1">Curso de machine learning de Andrew Ng</a>
      <a href="https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java">Ejemplo LeNet: MNIST Con Redes Convolucionales</a>
    </li>

  </ul>