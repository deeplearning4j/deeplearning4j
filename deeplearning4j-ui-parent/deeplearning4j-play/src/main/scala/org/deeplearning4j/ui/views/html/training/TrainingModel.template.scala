
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingModel_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingModel extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<div class="modelOuterDiv">
    <div class="modelGraphDiv" id="modelGraph">
        <p>View of network goes here - from the current FlowIterationListener</p>
        <p>User clicks on one of the nodes, and it loads the relevant data on the right</p>
    </div>
    <div class="modelContentDiv" id="modelLayerInfo">
        <p>
            Layer info table here
        </p>
        <p>
            Layer info available as a 2d array at /train/model/data/layerId -> "layerInfo"
        </p>
    </div>
    <div class="modelContentDiv" id="modelMeanMagnitudes">
        <p>Mean magnitudes multi line chart</p>
        <p>
            For now: multi-line chart of ratio of (mean(abs(updates))/mean(abs(parameters))) vs. time.
            One line for each parameter (weights, biases)
        </p>
        <p>
            /train/model/data/layerId -> "meanMagRatio"
            Contents:
            "layerParamNames": Names of the parameters for each parameter
            "iterCounts": iteration numbers (x axis values) for the ratio data points
            "someParamName": (key by contents of layerParamNames): y-axis values for ratios
        </p>
        <p>Maybe later, add the 'pre ratio' information: parameters mean magnitudes (1 per param type), updates mean magnitude (1 per param type)</p>
    </div>
    <div class="modelContentDiv" id="modelActivations">
        <p>Activations line chart</p>
        <p>Shows mean activations value over time +/- 2 standard deviations</p>
        <p>
            /train/model/data/layerId -> "activations"
            Properties:
            "iterCount": iterations at which the corresponding mean/stdev values occur (x axis values)
            "mean": mean values (y axis values)
            "stdev": standard deviation values
            Note that the mean + 2*stdev and mean - 2*stdev need to be calculated from these
        </p>
    </div>
    <div class="modelContentDiv" id="modelLearningRates">
        <p>Learning rates multi-line chart</p>
        <p>Shows learning rates (by parameter type) over time</p>
        <p>
            /train/model/data/layerId -> "learningRates"
            "paramNames" - names of parameters for the layer (order will be consistent between calls)
            "iterCounts" - x axis values
            "lrs": map with array of learning rate values, one per paramName. y-axis values
        </p>
    </div>
    <div class="modelContentDiv" id="modelParamHistogram">
        <p>Parameters histogram</p>
        <p>Histogram of the parameters in the network</p>
        <p>
            /train/model/data/layerId -> "paramHist"
            "paramNames": names of parameters
            one key for each param name: min,max,bins,counts fields. This defines the histogram content.
            i.e., each histogram has "bins" entries of size (max-min)/bins.
        </p>
    </div>
    <div class="modelContentDiv" id="modelParamHistogram">
        <p>Updates histogram</p>
        <p>Histogram of the updates in the network</p>
        <p>
            As per params histograms, but "updateHist"
        </p>
    </div>
</div>"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object TrainingModel extends TrainingModel_Scope0.TrainingModel
              /*
                  -- GENERATED --
                  DATE: Mon Oct 24 17:43:11 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: ed881b56c2e3c64aa44d53af9c02b046f33f6e3f
                  MATRIX: 598->1|731->39|759->41
                  LINES: 20->1|25->1|26->2
                  -- GENERATED --
              */
          