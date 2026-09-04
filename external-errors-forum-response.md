Thanks for posting the concrete example. There are two separate issues here: what `backpropGradient` expects, and whether several forward passes are supposed to precede one backward pass.

## First: why are there several forward passes before one backward pass?

I am not yet clear on the objective that requires that sequence. In ordinary backpropagation, one forward pass produces the activations needed by its corresponding backward pass. `MultiLayerNetwork` stores layer inputs from the **most recent** forward pass; another forward pass can replace that state. Consequently, several calls to `feedForward(...)` followed by one call to `backpropGradient(...)` do not backpropagate through all of those forward passes. The backward pass applies only to the retained, most recent activations.

If the intention is gradient accumulation over several examples or microbatches, the usual sequence is:

1. forward pass for microbatch 1;
2. backward pass for microbatch 1 and accumulate its raw gradient;
3. forward pass for microbatch 2;
4. backward pass for microbatch 2 and accumulate its raw gradient;
5. apply the updater once to the combined gradient.

That is several **forward/backward pairs before one parameter update**, not several forward passes before one backward pass. If the loss jointly depends on several network evaluations, such as a Siamese or contrastive objective, that relationship should normally be represented explicitly with a `ComputationGraph`, SameDiff, or a custom loss/layer. Could you clarify which of these cases you need?

## What “external error” means here

The example is unfortunately easy to misread. It is only demonstrating the mechanics of injecting an upstream gradient. The random array named `externalError` has no relationship to a target or to the model output, so that example is **not expected to converge**. A random external gradient is valid as an API demonstration, but not as a learning objective.

More importantly, `backpropGradient(epsilon, ...)` expects `epsilon` to be the derivative of the scalar loss with respect to the network's final activation:

\[
\epsilon = \frac{\partial L}{\partial a}
\]

It does not expect a scalar loss, and “error” is too ambiguous here. It might mean `target - prediction`, `prediction - target`, or an already differentiated loss. Those are not interchangeable.

For MSE,

\[
L = \operatorname{mean}((a-y)^2)
\]

and, before minibatch averaging,

\[
\frac{\partial L}{\partial a} = \frac{2(a-y)}{nOut}.
\]

The sign is important. Because the example ultimately performs

```java
model.params().subi(updateVector);
```

use `prediction - target`, not `target - prediction`. Reversing that sign performs gradient ascent and makes the loss increase.

## Do not multiply by the activation

`epsilon = error .* activation` is not the general correction. A `DenseLayer` already invokes its activation function's backpropagation internally. For a tanh output, the chain is:

\[
\frac{\partial L}{\partial a} = \frac{2(a-y)}{nOut},
\qquad
\frac{\partial L}{\partial z} =
\frac{\partial L}{\partial a}(1-a^2).
\]

The caller supplies the first expression; DL4J calculates the second. Multiplying by the activation itself is not the tanh derivative, and manually multiplying by an activation derivative would apply it twice.

The extra activation factor also explains a possible collapse toward zero: if the supplied value contains a factor of `a`, then `a = 0` can become a zero-gradient point even when the target is `0.4`.

## Minimal external-MSE training loop

Here is the essential pattern for a network whose last layer is a regular `DenseLayer`, not an `OutputLayer`. An identity output activation keeps the example easy to inspect:

```java
int minibatch = 32;
int nOut = 1;

MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(12345)
        .updater(new Sgd(0.05))
        .list()
        .layer(new DenseLayer.Builder()
                .nIn(1)
                .nOut(nOut)
                .activation(Activation.IDENTITY)
                .build())
        .build();

MultiLayerNetwork model = new MultiLayerNetwork(conf);
model.init();

INDArray input = Nd4j.ones(minibatch, 1);
INDArray target = Nd4j.valueArrayOf(new long[]{minibatch, nOut}, 0.4);

for (int iteration = 0; iteration < 200; iteration++) {
    model.setInput(input);

    // false means retain layer inputs because backprop needs this pass's activations.
    List<INDArray> activations = model.feedForward(true, false);
    INDArray prediction = activations.get(activations.size() - 1);

    INDArray residual = prediction.sub(target);       // prediction - target
    double mse = residual.mul(residual).meanNumber().doubleValue();

    // d(MSE)/d(output). Do not multiply by the activation derivative here.
    // The updater performs minibatch averaging using the minibatch argument below.
    INDArray epsilon = residual.mul(2.0 / nOut);

    Gradient gradient = model
            .backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces())
            .getFirst();

    // The updater modifies the Gradient in place: learning rate, momentum, Adam, etc.
    model.getUpdater().update(
            model,
            gradient,
            iteration,
            0,
            minibatch,
            LayerWorkspaceMgr.noWorkspaces());

    model.params().subi(gradient.gradient());
    model.setIterationCount(iteration + 1);
    model.clearLayersStates();

    if (iteration % 20 == 0) {
        System.out.println("iteration=" + iteration + ", mse=" + mse);
    }
}

System.out.println(model.output(input, false).getDouble(0));
```

The same `epsilon = 2 * (prediction - target) / nOut` rule applies if the final `DenseLayer` uses tanh; the layer applies the tanh derivative itself. Naturally, the target must then be in tanh's output range.

## State between iterations and epochs

There is no required `gradient.clear()` call. Backpropagation writes the next gradient into the model's gradient views, and `getUpdater().update(...)` intentionally modifies that gradient in place before it is subtracted from the parameters. Clearing the returned `Gradient` is unnecessary and can remove the variable-to-gradient mappings needed by the updater.

The documentation example hardcodes `iteration = 0` and `epoch = 0` because it performs exactly one update. Copying those constants unchanged into a training loop is incorrect, particularly with Adam/Nadam or scheduled learning rates.

The important bookkeeping points in a manual loop are:

- increment `iteration` on every parameter update, especially for Adam/Nadam and learning-rate schedules;
- increment `epoch` only after a complete pass over the training data;
- pass the actual minibatch size to the updater, because it performs minibatch averaging;
- retain activations for the matching backward pass with `feedForward(true, false)`;
- optionally clear layer state **after** backward/update to release references before the next iteration;
- do not clear the updater state between epochs—momentum/Adam state is supposed to persist.

So the original line passing an arbitrary random array is not useful for demonstrating convergence, but the underlying API does not require `error .* activation`. It requires the correctly signed and scaled upstream derivative, `dL/d(final activation)`.
