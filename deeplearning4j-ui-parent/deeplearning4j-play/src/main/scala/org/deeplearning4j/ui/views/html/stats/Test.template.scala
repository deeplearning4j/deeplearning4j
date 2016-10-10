
package org.deeplearning4j.ui.views.html.stats

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Test_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Test extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.4*/("""
"""),format.raw/*2.1*/("""<html>
<head>
<meta charset="utf-8"/>
    <title>Weights/Gradients/Score</title>
<style>
    .bar rect """),format.raw/*7.15*/("""{"""),format.raw/*7.16*/("""
        """),format.raw/*8.9*/("""fill: steelblue;
        shape-rendering: crispEdges;
    """),format.raw/*10.5*/("""}"""),format.raw/*10.6*/("""

    """),format.raw/*12.5*/(""".bar text """),format.raw/*12.15*/("""{"""),format.raw/*12.16*/("""
        """),format.raw/*13.9*/("""fill: #EFEFEF;
    """),format.raw/*14.5*/("""}"""),format.raw/*14.6*/("""

    """),format.raw/*16.5*/(""".area """),format.raw/*16.11*/("""{"""),format.raw/*16.12*/("""
        """),format.raw/*17.9*/("""fill: steelblue;
    """),format.raw/*18.5*/("""}"""),format.raw/*18.6*/("""

    """),format.raw/*20.5*/(""".axis path, .axis line """),format.raw/*20.28*/("""{"""),format.raw/*20.29*/("""
        """),format.raw/*21.9*/("""fill: none;
        stroke: #000;
        shape-rendering: crispEdges;
    """),format.raw/*24.5*/("""}"""),format.raw/*24.6*/("""

    """),format.raw/*26.5*/(""".tick line """),format.raw/*26.16*/("""{"""),format.raw/*26.17*/("""
        """),format.raw/*27.9*/("""opacity: 0.2;
        shape-rendering: crispEdges;
    """),format.raw/*29.5*/("""}"""),format.raw/*29.6*/("""

    """),format.raw/*31.5*/("""path """),format.raw/*31.10*/("""{"""),format.raw/*31.11*/("""
        """),format.raw/*32.9*/("""stroke: steelblue;
        stroke-width: 2;
        fill: none;
    """),format.raw/*35.5*/("""}"""),format.raw/*35.6*/("""

    """),format.raw/*37.5*/(""".legend """),format.raw/*37.13*/("""{"""),format.raw/*37.14*/("""
        """),format.raw/*38.9*/("""font-size: 12px;
        text-anchor: middle;
    """),format.raw/*40.5*/("""}"""),format.raw/*40.6*/("""

    """),format.raw/*42.5*/(""".brush .extent """),format.raw/*42.20*/("""{"""),format.raw/*42.21*/("""
        """),format.raw/*43.9*/("""stroke: #fff;
        fill-opacity: .125;
        shape-rendering: crispEdges;
    """),format.raw/*46.5*/("""}"""),format.raw/*46.6*/("""

"""),format.raw/*48.1*/("""</style>

    <!-- jQuery -->
    <script src="/assets/jquery-2.2.0.min.js"></script>

    <link href='/assets/roboto.css' rel='stylesheet' type='text/css'>

    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="/assets/bootstrap.min.css" />

    <!-- Optional theme -->
    <link rel="stylesheet" href="/assets/bootstrap-theme.min.css" />

    <!-- Latest compiled and minified JavaScript -->
    <script src="/assets/bootstrap.min.js" ></script>

    <!-- d3 -->
    <script src="/assets/d3.v3.min.js" charset="utf-8"></script>

    <script src="/assets/jquery-fileupload.js"></script>

    <!-- Booststrap Notify plugin-->
    <script src="/assets/bootstrap-notify.min.js"></script>

    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="/assets/bootstrap-select.min.css" />

    <!-- Latest compiled and minified JavaScript -->
    <script src="/assets/bootstrap-select.min.js"></script>

    <!-- DateTime formatter-->
    <script src="/assets/DateTimeFormat.js"></script>

    <script src="/assets/renderWeightsProper.js"></script>

    <script src="/assets/common.js"></script>

    <style>
        body """),format.raw/*86.14*/("""{"""),format.raw/*86.15*/("""
        """),format.raw/*87.9*/("""font-family: 'Roboto', sans-serif;
        color: #333;
        font-weight: 300;
        font-size: 16px;
        """),format.raw/*91.9*/("""}"""),format.raw/*91.10*/("""
        """),format.raw/*92.9*/(""".hd """),format.raw/*92.13*/("""{"""),format.raw/*92.14*/("""
        """),format.raw/*93.9*/("""background-color: #000000;
        font-size: 18px;
        color: #FFFFFF;
        """),format.raw/*96.9*/("""}"""),format.raw/*96.10*/("""
        """),format.raw/*97.9*/(""".block """),format.raw/*97.16*/("""{"""),format.raw/*97.17*/("""
        """),format.raw/*98.9*/("""width: 250px;
        height: 350px;
        display: inline-block;
        border: 1px solid #DEDEDE;
        margin-right: 64px;
        """),format.raw/*103.9*/("""}"""),format.raw/*103.10*/("""
        """),format.raw/*104.9*/(""".hd-small """),format.raw/*104.19*/("""{"""),format.raw/*104.20*/("""
        """),format.raw/*105.9*/("""background-color: #000000;
        font-size: 14px;
        color: #FFFFFF;
        """),format.raw/*108.9*/("""}"""),format.raw/*108.10*/("""
        """),format.raw/*109.9*/(""".charts, .chart """),format.raw/*109.25*/("""{"""),format.raw/*109.26*/("""
            """),format.raw/*110.13*/("""font-size: 10px;
            font-color: #000000;
            position: relative;
        """),format.raw/*113.9*/("""}"""),format.raw/*113.10*/("""
        """),format.raw/*114.9*/(""".scoreboard """),format.raw/*114.21*/("""{"""),format.raw/*114.22*/("""
            """),format.raw/*115.13*/("""position: absolute;
            top:  20px;
            right: 10px;
            z-index: 1000;
            font-size: 11px;
        """),format.raw/*120.9*/("""}"""),format.raw/*120.10*/("""
        """),format.raw/*121.9*/(""".score """),format.raw/*121.16*/("""{"""),format.raw/*121.17*/("""
            """),format.raw/*122.13*/("""font-size: 11px;
        """),format.raw/*123.9*/("""}"""),format.raw/*123.10*/("""
    """),format.raw/*124.5*/("""</style>


</head>
<body>
<table style="width: 100%; padding: 5px;" class="hd">
    <tbody>
    <tr>
        <td style="width: 48px;"><a href="/"><img src="/assets/deeplearning4j.img" border="0"/></a></td>
        <td>DeepLearning4j UI</td>
        <td style="width: 512px; text-align: right;" class="hd-small">&nbsp; Available sessions: <select class="selectpicker" id="sessionSelector" onchange="window.location.href = 'weights?sid='+ this.options[this.selectedIndex].value ;" style="color: #000000; display: inline-block; width: 256px;">
            <option value="0" selected="selected">Pick a session to track</option>
        </select>&nbsp;&nbsp;
<script>
    buildSessionSelector2("weights","listSessions");
</script>
        </td>
        <td style="width: 256px;" class="hd-small">&nbsp;Updated at: <b><span id="updatetime">No updates so far</span></b>&nbsp;</td>
    </tr>
    </tbody>
</table>

<div style="width: 100%; text-align: center;">
    <div id="display" style="width: 1540px; height: 900px; text-align: left; background-color: #FFFFFF; display: inline-block; overflow: hidden; ">
        <div id="scores" style="background-color: #EFEFEF; display: block; float: left; width: 750px; height: 400px; border: 1px solid #CECECE; margin: 10px;">
            <h5>&nbsp;&nbsp;Score vs. Iteration #</h5>
            <div class="chart" id="schart">
            </div>
        </div>
        <div id="model" style="position: relative; background-color: #EFEFEF; display: block; float: left; width: 750px; height: 400px; border: 1px solid #CECECE; margin: 10px;">
            <h5>&nbsp;&nbsp;Model</h5>
            <div class="charts"></div>
            <div style="position: absolute; top: 5px; right: 5px;">
                <select id="modelSelector" onchange="selectModel();">
                </select>
            </div>
        </div>
        <div id="gradient" style="position: relative; background-color: #EFEFEF; display: block; float: left; width: 750px; height: 400px; border: 1px solid #CECECE; margin: 10px;">
            <h5>&nbsp;&nbsp;Gradient</h5>
            <div class="charts"></div>
            <div style="position: absolute; top: 5px; right: 5px;">
                <select id="gradientSelector" onchange="selectGradient();">
                </select>
            </div>
        </div>
        <div id="magnitudes" style="position: relative; background-color: #EFEFEF; display: block; float: left; width: 750px; height: 400px; border: 1px solid #CECECE; margin: 10px;">
            <h5>&nbsp;&nbsp;Mean Magnitudes: Parameters and Updates</h5>
            <div class="charts"></div>
            <div style="position: absolute; top: 5px; right: 5px;">
                <select id="magnitudeSelector" onchange="selectMagnitude();">
                </select>
            </div>
        </div>
        <!--<div id="lastupdate">
            <div class="updatetime">-1</div>
        </div>-->
    </div>

    <!--
    <div style="display: block;">
        nav bar
    </div> -->
</div>
<!--
<div id="score" style="display: inline-block; width: 650px; height: 400px; border: 1px solid #CECECE;">
    <h4>Score</h4>
    <div class="score"></div>
</div>-->

</body>
</html>"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object Test extends Test_Scope0.Test
              /*
                  -- GENERATED --
                  DATE: Mon Oct 10 12:12:03 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/stats/Test.scala.html
                  HASH: e64e8f7c3ee3d060cf2315b9c48d449a1844033d
                  MATRIX: 546->1|642->3|670->5|805->113|833->114|869->124|956->184|984->185|1019->193|1057->203|1086->204|1123->214|1170->234|1198->235|1233->243|1267->249|1296->250|1333->260|1382->282|1410->283|1445->291|1496->314|1525->315|1562->325|1667->403|1695->404|1730->412|1769->423|1798->424|1835->434|1919->491|1947->492|1982->500|2015->505|2044->506|2081->516|2179->587|2207->588|2242->596|2278->604|2307->605|2344->615|2423->667|2451->668|2486->676|2529->691|2558->692|2595->702|2708->788|2736->789|2767->793|3994->1992|4023->1993|4060->2003|4206->2122|4235->2123|4272->2133|4304->2137|4333->2138|4370->2148|4484->2235|4513->2236|4550->2246|4585->2253|4614->2254|4651->2264|4823->2408|4853->2409|4891->2419|4930->2429|4960->2430|4998->2440|5113->2527|5143->2528|5181->2538|5226->2554|5256->2555|5299->2569|5420->2662|5450->2663|5488->2673|5529->2685|5559->2686|5602->2700|5768->2838|5798->2839|5836->2849|5872->2856|5902->2857|5945->2871|5999->2897|6029->2898|6063->2904
                  LINES: 20->1|25->1|26->2|31->7|31->7|32->8|34->10|34->10|36->12|36->12|36->12|37->13|38->14|38->14|40->16|40->16|40->16|41->17|42->18|42->18|44->20|44->20|44->20|45->21|48->24|48->24|50->26|50->26|50->26|51->27|53->29|53->29|55->31|55->31|55->31|56->32|59->35|59->35|61->37|61->37|61->37|62->38|64->40|64->40|66->42|66->42|66->42|67->43|70->46|70->46|72->48|110->86|110->86|111->87|115->91|115->91|116->92|116->92|116->92|117->93|120->96|120->96|121->97|121->97|121->97|122->98|127->103|127->103|128->104|128->104|128->104|129->105|132->108|132->108|133->109|133->109|133->109|134->110|137->113|137->113|138->114|138->114|138->114|139->115|144->120|144->120|145->121|145->121|145->121|146->122|147->123|147->123|148->124
                  -- GENERATED --
              */
          