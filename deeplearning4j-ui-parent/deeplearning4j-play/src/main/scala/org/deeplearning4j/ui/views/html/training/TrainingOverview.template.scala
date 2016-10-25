
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingOverview_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingOverview extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<div class="overviewOuterDiv">
    <div class="overviewScoreChartDiv" id="scoreChartDiv">
        <p>Score chart here</p>
        <p>Data from: train/overview/data</p>
        <p>x-axis values: "scoresIter"</p>
        <p>y-axis values: "scores"</p>
    </div>
    <div class="overviewTableDiv">
        <p>Performance and Model Info Here</p>
        <p>This will change every iteration</p>
        <p>Source: /train/overview/data -> "perf" and "model"</p>
        <table class="overviewTable">
            <tr>
                <td>Performance</td>
                <td>data</td>
            </tr>
            <tr>
                <td>goes</td>
                <td>here</td>
            </tr>
        </table>
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
object TrainingOverview extends TrainingOverview_Scope0.TrainingOverview
              /*
                  -- GENERATED --
                  DATE: Tue Oct 25 20:32:36 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 5c570e1de495df0d7eac32bb009c90e8437c530f
                  MATRIX: 604->1|737->39|765->41
                  LINES: 20->1|25->1|26->2
                  -- GENERATED --
              */
          