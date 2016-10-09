
package org.deeplearning4j.ui.views.histogram

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Histogram_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Histogram extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

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
    buildSessionSelector("HISTOGRAM");
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
object Histogram extends Histogram_Scope0.Histogram
              /*
                  -- GENERATED --
                  DATE: Mon Oct 10 09:54:46 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/histogram/Histogram.scala.html
                  HASH: e64c1f9f85467f0891c3248c9a0e1a26b8152f01
                  MATRIX: 560->1|656->3|684->5|819->113|847->114|883->124|970->184|998->185|1033->193|1071->203|1100->204|1137->214|1184->234|1212->235|1247->243|1281->249|1310->250|1347->260|1396->282|1424->283|1459->291|1510->314|1539->315|1576->325|1681->403|1709->404|1744->412|1783->423|1812->424|1849->434|1933->491|1961->492|1996->500|2029->505|2058->506|2095->516|2193->587|2221->588|2256->596|2292->604|2321->605|2358->615|2437->667|2465->668|2500->676|2543->691|2572->692|2609->702|2722->788|2750->789|2781->793|4008->1992|4037->1993|4074->2003|4220->2122|4249->2123|4286->2133|4318->2137|4347->2138|4384->2148|4498->2235|4527->2236|4564->2246|4599->2253|4628->2254|4665->2264|4837->2408|4867->2409|4905->2419|4944->2429|4974->2430|5012->2440|5127->2527|5157->2528|5195->2538|5240->2554|5270->2555|5313->2569|5434->2662|5464->2663|5502->2673|5543->2685|5573->2686|5616->2700|5782->2838|5812->2839|5850->2849|5886->2856|5916->2857|5959->2871|6013->2897|6043->2898|6077->2904
                  LINES: 20->1|25->1|26->2|31->7|31->7|32->8|34->10|34->10|36->12|36->12|36->12|37->13|38->14|38->14|40->16|40->16|40->16|41->17|42->18|42->18|44->20|44->20|44->20|45->21|48->24|48->24|50->26|50->26|50->26|51->27|53->29|53->29|55->31|55->31|55->31|56->32|59->35|59->35|61->37|61->37|61->37|62->38|64->40|64->40|66->42|66->42|66->42|67->43|70->46|70->46|72->48|110->86|110->86|111->87|115->91|115->91|116->92|116->92|116->92|117->93|120->96|120->96|121->97|121->97|121->97|122->98|127->103|127->103|128->104|128->104|128->104|129->105|132->108|132->108|133->109|133->109|133->109|134->110|137->113|137->113|138->114|138->114|138->114|139->115|144->120|144->120|145->121|145->121|145->121|146->122|147->123|147->123|148->124
                  -- GENERATED --
              */
          