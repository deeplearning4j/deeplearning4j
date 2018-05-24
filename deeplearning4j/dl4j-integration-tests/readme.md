
#DL4J Integration Tests

These tests are designed to check a number of aspects of DL4J:
1. Predictions
2. Training (training curves, parameters, gradient calculation)
3. Evaluation
4. Model serialization
5. Overfitting sanity checks
6. Data pipelines
7. Evaluation classes
8. Parallel Wrapper
9. Validating conditions that should always hold (frozen layer params don't change, for example)


They are designed for the following purposes:
1. Detecting regressions: i.e., new commit changed or broke previously working functionality
2. Detecting integration issues - i.e., issues that show up only when components are used together (but not in isolation in unit test)
3. Detecting significant differences between CPU and CUDA backends
4. Validating implementation via sanity checks on training - i.e., can we overfit a single example?
5. Checking networks and data pipelines on real-world scale data and nets
6. Operating as fully automated pre-release checks (replacing previously used manual checks)

## Types of Tests

The integration tests are set up to be able to run multiple tests on each network configuration.

Networks may be pretrained (from model zoo) or randomly initialized (from specified configuration).

Specifically, test cases can be run with any subset of the following components to be tested, by setting TestCase.XYZ boolean options to true or false:

1. testPredictions: Testing output (predictions) on some specified data vs. saved/known good arrays
2. testGradients: Testing gradients on some specified data vs. saved/known good arrays
3. testPretrain: Test layerwise pretraining parameters and training curves
4. testTrainingCurves: Train, and check score vs. iteration
5. testParamsPostTraining: validate params match post training
6. testEvaluation: test the evaluation performance (post training, if 4 or 5 are true)
7. testParallelInference: validate that single net and parallel inference results match
8. testOverfitting: sanity check - try to overfit a single example



## Adding a New Integration Test

The process to add a new test is simple:
1. Add a method that creates and returns a TestCase object
2. Add it as a unit test to IntegrationTests class
3. Run IntegrationTestBaselineGenerator (if required) to generate and save the "known good" results.

Note that IntegrationTestBaselineGenerator assumes you have the dl4j-test-resources cloned parallel to the DL4J mono-repo.