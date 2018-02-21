---
title: Machine Learning Workflows in Production
layout: default
redirect: workflow
---

# Machine Learning Workflows in Production

Machine learning in production happens in five phases. (There are few standardized best practices across teams and companies in the industry. Most machine-learning systems are ad hoc.)

### Phases in Machine Learning Workflows

* Use Case Conception and Formulation
* Feasibility Study and Exploratory Analysis
* Model Design, Training, and Offline Evaluation
* Model Deployment, Online Evaluation, and Monitoring
* Model Maintenance, Diagnosis, and Retraining

Within each phase, we'll explain: 

* What specific tasks are performed?
* Who is involved in each phase (businessperson, data scientist/engineer, DevOps)?

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH PRODUCTION ML</a>
</p>

### Relevant Personnel and Roles

* **Decision Maker**: holds the purse strings, can wrangle funding and resources (might be same as Stakeholder).
* **Stakeholder**: businessperson who cares about problem, who can state/quantify the business value of potential solutions.
* **Domain Expert**: person who understands the domain and problem, may also know about the data (might be the same as Stakeholder).
* **Data Scientist**: ML expert who can turn business problem into a well-defined ML task and propose one or more possible approaches.
* **Data Engineer**: database admin (DBA) or similar who knows where data lives, can comment on its size and contents (might be same as Data Scientist).
* **Systems Architect / DevOps**: systems architect or similar who is expert on big data and production software infrastructure, deployment, etc.

## Phase 1: Use Case Conception and Formulation

**Goal** 
Identify a data-intensive business problem and propose a potential machine learning solution.

**Tasks**
* Identify use case, define business value (labor/cost savings, fraud prevention and reduction, increased clickthrough rate, etc.)
* Re-state business problem as machine learning task, e.g., anomaly detection or classification
* Define "success" -- choose metric, e.g., AUC, and minimum acceptable performance, quantify potential business value
* Identify relevant and necessary data and available data sources
* Quick and dirty literature review
* Define necessary system architecture
* Assess potential sources of risk
* Commission exploratory analysis or feasibility study, if appropriate

**People**
* Critical: Decision Maker, Stakeholder, Data Scientist
* Other: Domain Expert (if Stakeholder doesn't know problem), Data Engineer (if DS doesn't know data systems), Systems Architect (if discussing deployment)


## Phase 2: Feasibility Study and Exploratory Analysis

**Goal**
Rapidly explore and de-risk a use case before significant engineering resources are dedicated to it, make "go/no go" recommendation

*NOTE: overlaps with Phase 3 (model training) except that here you don't expect a fully tuned model, nor do you expect to produce a reusable software artifact.*

**Tasks**
* Exploratory data analysis (EDA): descriptive statistics, visualization, detection of garbage data/noise/outlier values, quantify signal-to-noise ratio
* Quantify suitability of data for ML: number of records and features, availability and quality of labels, 
* Specify experimental (i.e. training/test split) protocol
* Rapid data ETL (extract, transform, and load) and vectorization to build experimental data sets (which might be only a toy subset)
* Thorough literature review with short list of proposed machine-learning approaches
* Train and evaluate ML models to assess presence (or absence) of predictive signal
* Make "go/no go" recommendation

**People**
* Data Engineer, Data Scientist: explore data, run experiments, produce reports
* Stakeholder, Domain Expert: answer questions, as needed
* Decision Maker, Stakeholder: consume final report/recommendation

<!-- SKIL Support: ETL, simple EDA, model training, and evaluation are supported by Workspaces/Experiments/notebooks -->

<p align="center">
<a href="https://skymind.ai/services" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">SKYMIND PROOF-OF-CONCEPT PROJECTS</a></br>
<a href="https://deeplearning4j.org/datavec" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">ETL FOR MACHINE LEARNING</a></br>
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">NOTEBOOKS FOR INTERACTIVE DATA EXPLORATION</a>
</p>

## Phase 3: Model Design, Training, and Offline Evaluation

**Goals**
* Train best performing model possible given available data, computational resources, and time.
* Build reliable, reusable software pipeline for re-training models in the future.

*NOTE: overlaps with Phase 2 (feasibility study), but here you expect a fully tuned model and a reusable software artifact.*

**Tasks**
* Plan full set of experiments
* Data ETL and vectorization pipeline that is configurable, fully tested, scalable, automatable
* Model training code that is configurable, fully tested, scalable, automatable
* "Offline" (on held-out, not live, data) model evaluation code that is configurable, fully tested, scalable, automatable
* Design, train, and evaluate models
* Tune and debug model training
* Thorough empirical comparison of competing models, hyperparameters
* Document experiments and model performance to date
* Save deployable artifacts (transforms, models, etc.)

**People**
* Data Engineer: ETL, assist DS with infrastructure as needed
* Data Scientist: plan and execute model training and evaluation, produce "reports" (automated by tools)
* Stakeholder, Domain Expert: answer questions, as needed; consume "reports" on progress/performance; provide feedback
* Decision Maker, Stakeholder: consume "reports" on progress/performance

<p align="center">
<a href="https://skymind.ai/services" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">SKYMIND PROFESSIONAL SERVICES</a>
<a href="https://deeplearning4j.org/spark" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">SCALING UP TRAINING ON APACHE SPARK</a>
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">MANAGING AND TRACKING TRAINED MODELS</a>
</p>

## Phase 4: Model Deployment, Online Evaluation, and Monitoring

**Goals**
* Deploy trained model (and transform, if needed) as service, integrate with other software/processes
* Monitor and log deployed model status, performance, and accuracy

**Tasks**
* Deploy models (and transforms) as consumable software services via, e.g., REST API
* Plan and execute trial deployments and experiments, e.g., A/B tests to compare new vs. old models
* Deploy to controlled staging environment, measure performance and accuracy on live data but don't expose
* Log and detect errors in deployment, e.g.:
    * Transform fails because schema does not match live data
    * Model fails due to invalid vectorized data input size
    * Transform or model servers die or become unreachable
* Log and track model performance and accuracy on live data, look for:
    * Poor prediction throughput (might need to add more servers)
    * Model drift, i.e., gradual decline in accuracy (might need to retrain model on more recent data)
    * Unexpected poor accuracy (might need to roll back model)

**People**
* "Gatekeeper:" someone or some group of people responsible for "blessing" models, i.e., deciding whether and which a model should go live (probably Decision Maker or Stakeholder with advice and consent of Data Scientist and Dev Ops)
* System Architect: deploy models, manage monitor model status and performance
* Data Scientist: plan A/B tests (or other trial deployments), consume reports on model accuracy
* Stakeholder, Domain Expert: answer questions, as needed; consume reports on model accuracy, provide feedback

SKIL Support: one-click deployment of trained or imported models, simple monitoring of model status

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">DEPLOY ML WITH ONE CLICK</a>
</p>

## Phase 5: Model Maintenance, Diagnosis, and Retraining

**Goals**
* Monitor and log deployed model accuracy over longer periods of time
* Gather statistics on deployed models to feed back into training and deployment

**Tasks**
* Gather statistics on deployed models, such as how long it takes for deployed models to become "stale" (i.e., accuracy on live data drops below acceptable threshold); Patterns in model inaccuracies (might need to re-design model architecture to account for new feature or to correct faulty assumption)
* Formulate new hypotheses or experiments based on insights from tracking performance

**People**
* System Architect: monitor model status and performance
* Data Scientist: consume reports on model accuracy
* Stakeholder, Domain Expert: answer questions, as needed; consume reports on model accuracy, provide feedback

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">MONITORING AND LOGGING FOR DEPLOYED ML</a>
</p>

## <a name="intro">Other Machine Learning Tutorials</a>

For people just getting started with deep learning, the following tutorials and videos provide an easy entrance to the fundamental ideas of feedforward networks:

* [Recurrent Networks and LSTMs](./lstm.html)
* [Deep Reinforcement Learning](./deepreinforcementlearning.html)
* [Deep Convolutional Networks](./convolutionalnets.html)
* [Multilayer Perceptron (MLPs) for Classification](./multilayerperceptron.html)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network.html)
* [Symbolic Reasoning & Deep Learning](./symbolicreasoning.html)
* [Using Graph Data with Deep Learning](./graphdata.html)
* [AI vs. Machine Learning vs. Deep Learning](./ai-machinelearning-deeplearning.html)
* [Markov Chain Monte Carlo & Machine Learning](/markovchainmontecarlo.html)
* [MNIST for Beginners](./mnist-for-beginners.html)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine.html)
* [Eigenvectors, PCA, Covariance and Entropy](./eigenvector.html)
* [Glossary of Deep-Learning and Neural-Net Terms](./glossary.html)
* [Word2vec and Natural-Language Processing](./word2vec.html)
* [Deeplearning4j Examples via Quickstart](./quickstart.html)
* [Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs) (A seven-video series)
* [Inference: Machine Learning Model Server](./modelserver.html)
