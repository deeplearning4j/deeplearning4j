
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingSystem_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingSystem extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<div class="systemOuterDiv">
    <div class="systemContentDiv" id="systemMemoryUtilization">
        <p>Memory utilization chart (0 to 100%, last N minutes?) goes here</p>
        <p>Note that in general for this page: will be showing stats/info from MULTIPLE systems</p>
        <p>Single machine training will have 1; Spark etc. training will have one row like this for each machine/JVM</p>
    </div>
    <div class="systemContentDiv" id="systemDeviceMemory">
        <p>Memory utilization for GPUs (0 to 100%, last N minutes?) goes here</p>
        <p>For machines without any GPUs/devices, we shouldn't show this div/chart</p>
    </div>
    <div class="systemContentDiv" id="systemHardwareInfo">
        <p>Hardware information table for this machine/JVM</p>
    </div>
    <div class="systemContentDiv" id="systemHardwareInfo">
        <p>Software information table for this machine/JVM</p>
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
object TrainingSystem extends TrainingSystem_Scope0.TrainingSystem
              /*
                  -- GENERATED --
                  DATE: Sun Oct 16 13:32:31 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: 2fa53382daa00608b8fd23e3492223dcb2f06955
                  MATRIX: 600->1|733->39|761->41
                  LINES: 20->1|25->1|26->2
                  -- GENERATED --
              */
          