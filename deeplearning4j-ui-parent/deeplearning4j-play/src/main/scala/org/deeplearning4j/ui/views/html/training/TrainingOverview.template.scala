
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
"""),format.raw/*2.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*7.17*/i18n/*7.21*/.getMessage("train.pagetitle")),format.raw/*7.51*/("""</title>
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

        <link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
        <link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        """),format.raw/*16.214*/("""
        """),format.raw/*17.9*/("""<link href='/assets/css/opensans-fonts.css' rel='stylesheet' type='text/css'>
        <link rel="shortcut icon" href="/assets/img/favicon.ico">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->

    </head>

    <body>
            <!-- Start Header -->
        <div class="navbar">
            <div class="navbar-inner">
                <div class="container-fluid">
                    <a class="btn btn-navbar" data-toggle="collapse" data-target=".top-nav.nav-collapse,.sidebar-nav.nav-collapse">
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </a>
                    <a class="brand" href="#"><span>"""),_display_(/*42.54*/i18n/*42.58*/.getMessage("train.pagetitle")),format.raw/*42.88*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        """),_display_(/*44.26*/i18n/*44.30*/.getMessage("train.session.label")),format.raw/*44.64*/("""
                        """),format.raw/*45.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    <div id="workerSelectDiv" style="display:none; float:right;">
                        """),_display_(/*50.26*/i18n/*50.30*/.getMessage("train.session.worker.label")),format.raw/*50.71*/("""
                        """),format.raw/*51.25*/("""<select id="workerSelect" onchange='selectNewWorker()'>
                            <option>(Worker ID)</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
            <!-- End Header -->

        <div class="container-fluid-full">
            <div class="row-fluid">

                    <!-- Start Main Menu -->
                <div id="sidebar-left" class="span2">
                    <div class="nav-collapse sidebar-nav">
                        <ul class="nav nav-tabs nav-stacked main-menu">
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">"""),_display_(/*67.137*/i18n/*67.141*/.getMessage("train.nav.overview")),format.raw/*67.174*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">"""),_display_(/*68.104*/i18n/*68.108*/.getMessage("train.nav.model")),format.raw/*68.138*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">"""),_display_(/*69.109*/i18n/*69.113*/.getMessage("train.nav.system")),format.raw/*69.144*/("""</span></a></li>
                            """),format.raw/*70.160*/("""
                            """),format.raw/*71.29*/("""<li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*72.146*/i18n/*72.150*/.getMessage("train.nav.language")),format.raw/*72.183*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('de', 'overview')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        Deutsch</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        український</span></a></li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                    <!-- End Main Menu -->

                <noscript>
                    <div class="alert alert-block span10">
                        <h4 class="alert-heading">Warning!</h4>
                        <p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">
                            JavaScript</a> enabled to use this site.</p>
                    </div>
                </noscript>

                    <!-- Start Score Chart-->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span8">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*110.41*/i18n/*110.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*110.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*114.55*/i18n/*114.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*114.110*/("""
                                    """),format.raw/*115.37*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*115.70*/i18n/*115.74*/.getMessage("train.overview.charts.iteration")),format.raw/*115.120*/("""
                                    """),format.raw/*116.37*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                            <!-- End Score Chart-->
                            <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*124.41*/i18n/*124.45*/.getMessage("train.overview.perftable.title")),format.raw/*124.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*129.46*/i18n/*129.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*129.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*133.46*/i18n/*133.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*133.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*137.46*/i18n/*137.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*137.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*141.46*/i18n/*141.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*141.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*145.46*/i18n/*145.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*145.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*149.46*/i18n/*149.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*149.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*153.46*/i18n/*153.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*153.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*157.46*/i18n/*157.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*157.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*161.46*/i18n/*161.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*161.104*/("""</td>
                                        <td id="examplesPerSec">Loading...</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                            <!--End Model Table -->
                    </div>


                    <div class="row-fluid">
                            <!--Start Ratio Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*175.41*/i18n/*175.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*175.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*179.55*/i18n/*179.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*179.116*/("""
                                    """),format.raw/*180.37*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*181.47*/i18n/*181.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*181.108*/("""
                                    """),format.raw/*182.37*/(""":</b> <span id="yLogRatio">0</span>
                                    , <b>"""),_display_(/*183.43*/i18n/*183.47*/.getMessage("train.overview.charts.iteration")),format.raw/*183.93*/(""":</b> <span id="xRatio">
                                        0</span></p>
                            </div>
                        </div>
                            <!--End Ratio Table -->
                            <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*191.41*/i18n/*191.45*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*191.91*/(""": log<sub>10</sub></b></h2>
                                <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -11px; right: 22px;">
                                    <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*193.156*/i18n/*193.160*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*193.216*/("""</a></li>
                                    <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*194.137*/i18n/*194.141*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*194.195*/("""</a></li>
                                    <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*195.133*/i18n/*195.137*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*195.189*/("""</a></li>
                                </ul>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*200.55*/i18n/*200.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*200.110*/("""
                                    """),format.raw/*201.37*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*202.47*/i18n/*202.51*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*202.102*/("""
                                    """),format.raw/*203.37*/(""":</b> <span id="yLogStdev">0</span>
                                    , <b>"""),_display_(/*204.43*/i18n/*204.47*/.getMessage("train.overview.charts.iteration")),format.raw/*204.93*/(""":</b> <span id="xStdev">
                                        0</span></p>
                            </div>
                        </div>
                            <!-- End Variance Table -->
                    </div>
                </div>
            </div><!-- End Content Span10-->
        </div><!--End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/js/jquery-1.9.1.min.js"></script>
        <script src="/assets/js/jquery-migrate-1.0.0.min.js"></script>
        <script src="/assets/js/jquery-ui-1.10.0.custom.min.js"></script>
        <script src="/assets/js/jquery.ui.touch-punch.js"></script>
        <script src="/assets/js/modernizr.js"></script>
        <script src="/assets/js/bootstrap.min.js"></script>
        <script src="/assets/js/jquery.cookie.js"></script>
        <script src="/assets/js/fullcalendar.min.js"></script>
        <script src="/assets/js/jquery.dataTables.min.js"></script>
        <script src="/assets/js/excanvas.js"></script>
        <script src="/assets/js/jquery.flot.js"></script>
        <script src="/assets/js/jquery.flot.pie.js"></script>
        <script src="/assets/js/jquery.flot.stack.js"></script>
        <script src="/assets/js/jquery.flot.resize.min.js"></script>
        <script src="/assets/js/jquery.flot.selection.js"></script>
        <script src="/assets/js/jquery.chosen.min.js"></script>
        <script src="/assets/js/jquery.uniform.min.js"></script>
        <script src="/assets/js/jquery.cleditor.min.js"></script>
        <script src="/assets/js/jquery.noty.js"></script>
        <script src="/assets/js/jquery.elfinder.min.js"></script>
        <script src="/assets/js/jquery.raty.min.js"></script>
        <script src="/assets/js/jquery.iphone.toggle.js"></script>
        <script src="/assets/js/jquery.uploadify-3.1.min.js"></script>
        <script src="/assets/js/jquery.gritter.min.js"></script>
        <script src="/assets/js/jquery.imagesloaded.js"></script>
        <script src="/assets/js/jquery.masonry.min.js"></script>
        <script src="/assets/js/jquery.knob.modified.js"></script>
        <script src="/assets/js/jquery.sparkline.min.js"></script>
        <script src="/assets/js/counter.js"></script>
        <script src="/assets/js/retina.js"></script>
        <script src="/assets/js/train/overview.js"></script>    <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*250.47*/("""{"""),format.raw/*250.48*/("""
                    """),format.raw/*251.21*/("""renderOverviewPage(true);
                """),format.raw/*252.17*/("""}"""),format.raw/*252.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*257.41*/("""{"""),format.raw/*257.42*/("""
                    """),format.raw/*258.21*/("""renderOverviewPage(false);
                """),format.raw/*259.17*/("""}"""),format.raw/*259.18*/(""", 2000);
        </script>
            <!-- End JavaScript-->

    </body>
</html>
"""))
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
                  DATE: Thu Feb 16 17:33:43 AEDT 2017
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 0a0816316a3d82ced7e1cad974f6c94e1ffd5f23
                  MATRIX: 604->1|737->39|765->41|888->138|900->142|950->172|1513->911|1550->921|2668->2012|2681->2016|2732->2046|2880->2167|2893->2171|2948->2205|3002->2231|3318->2520|3331->2524|3393->2565|3447->2591|4201->3317|4215->3321|4270->3354|4419->3475|4433->3479|4485->3509|4639->3635|4653->3639|4706->3670|4781->3847|4839->3877|5018->4028|5032->4032|5087->4065|7819->6769|7833->6773|7901->6819|8189->7079|8203->7083|8277->7134|8344->7172|8405->7205|8419->7209|8488->7255|8555->7293|8973->7683|8987->7687|9054->7732|9371->8021|9385->8025|9458->8075|9699->8288|9713->8292|9783->8340|10022->8551|10036->8555|10106->8603|10345->8814|10359->8818|10430->8867|10671->9080|10685->9084|10760->9136|11004->9352|11018->9356|11091->9406|11333->9620|11347->9624|11427->9681|11676->9902|11690->9906|11766->9959|12011->10176|12025->10180|12102->10234|12699->10803|12713->10807|12787->10859|13095->11139|13109->11143|13189->11200|13256->11238|13377->11331|13391->11335|13471->11392|13538->11430|13645->11509|13659->11513|13727->11559|14157->11961|14171->11965|14239->12011|14579->12322|14594->12326|14673->12382|14849->12529|14864->12533|14941->12587|15113->12730|15128->12734|15203->12786|15526->13081|15540->13085|15614->13136|15681->13174|15802->13267|15816->13271|15890->13322|15957->13360|16064->13439|16078->13443|16146->13489|18816->16130|18846->16131|18897->16153|18969->16196|18999->16197|19168->16337|19198->16338|19249->16360|19322->16404|19352->16405
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|40->16|41->17|66->42|66->42|66->42|68->44|68->44|68->44|69->45|74->50|74->50|74->50|75->51|91->67|91->67|91->67|92->68|92->68|92->68|93->69|93->69|93->69|94->70|95->71|96->72|96->72|96->72|134->110|134->110|134->110|138->114|138->114|138->114|139->115|139->115|139->115|139->115|140->116|148->124|148->124|148->124|153->129|153->129|153->129|157->133|157->133|157->133|161->137|161->137|161->137|165->141|165->141|165->141|169->145|169->145|169->145|173->149|173->149|173->149|177->153|177->153|177->153|181->157|181->157|181->157|185->161|185->161|185->161|199->175|199->175|199->175|203->179|203->179|203->179|204->180|205->181|205->181|205->181|206->182|207->183|207->183|207->183|215->191|215->191|215->191|217->193|217->193|217->193|218->194|218->194|218->194|219->195|219->195|219->195|224->200|224->200|224->200|225->201|226->202|226->202|226->202|227->203|228->204|228->204|228->204|274->250|274->250|275->251|276->252|276->252|281->257|281->257|282->258|283->259|283->259
                  -- GENERATED --
              */
          