
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
        <p>Mean magnitudes line chart</p>
        <p>Line chart of mean magnitudes</p>
        <p>Selection box with: ratio of params (default), parameters mean magnitudes (1 per param type), updates mean magnitude (1 per param type)</p>
    </div>
    <div class="modelContentDiv" id="modelActivations">
        <p>Activations line chart</p>
        <p>Shows mean activations value over time +/- 2 standard deviations</p>
    </div>
    <div class="modelContentDiv" id="modelLearningRates">
        <p>Learning rates line chart</p>
        <p>Shows learning rates (by parameter type) over time</p>
    </div>
    <div class="modelContentDiv" id="modelParamHistogram">
        <p>Parameters histogram</p>
        <p>Histogram of the parameters in the network</p>
    </div>
    <div class="modelContentDiv" id="modelParamHistogram">
        <p>Updates histogram</p>
        <p>Histogram of the updates in the network</p>
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
                  DATE: Sun Oct 16 15:55:40 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: 106663ac19fa95d88f93e00338d029a5476322eb
                  MATRIX: 598->1|731->39|759->41
                  LINES: 20->1|25->1|26->2
                  -- GENERATED --
              */
          