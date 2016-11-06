
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
        <link href='http://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800&subset=latin,cyrillic-ext,latin-ext' rel='stylesheet' type='text/css'>
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
                    <a class="brand" href="index.html"><span>"""),_display_(/*41.63*/i18n/*41.67*/.getMessage("train.pagetitle")),format.raw/*41.97*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        Session:
                        <select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
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
                            <li><a href="overview"><i class="icon-bar-chart"></i><span class="hidden-tablet"> """),_display_(/*60.112*/i18n/*60.116*/.getMessage("train.nav.overview")),format.raw/*60.149*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> """),_display_(/*61.105*/i18n/*61.109*/.getMessage("train.nav.model")),format.raw/*61.139*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-dashboard"></i><span class="hidden-tablet"> """),_display_(/*62.138*/i18n/*62.142*/.getMessage("train.nav.system")),format.raw/*62.173*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet"> """),_display_(/*63.103*/i18n/*63.107*/.getMessage("train.nav.userguide")),format.raw/*63.141*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">
                                    Language</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> український</span></a></li>
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

                    <!-- Start Content -->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span12">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*96.41*/i18n/*96.45*/.getMessage("train.system.title")),format.raw/*96.78*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <ul class="nav tab-menu nav-tabs" id="systemTab"></ul>

                                    <!--Start System Tab -->
                                <div id="myTabContent" class="tab-content">
                                    <div class="tab-pane active">

                                            <!-- System Memory Utilization Chart -->
                                        <div class="row-fluid">

                                            <div class="box span12" id="systemMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*110.61*/i18n/*110.65*/.getMessage("train.system.chart.systemTitle")),format.raw/*110.110*/(""" """),format.raw/*110.111*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="systemMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*114.75*/i18n/*114.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*114.124*/(""":</b> <span id="y">0</span>, <b>
                                                        """),_display_(/*115.58*/i18n/*115.62*/.getMessage("train.overview.charts.iteration")),format.raw/*115.108*/(""":</b> <span id="x">0</span></p>
                                                </div>
                                            </div>

                                            <!-- GPU Memory Utlization Chart -->
                                            <div class="box span6" id="gpuMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*122.61*/i18n/*122.65*/.getMessage("train.system.chart.gpuTitle")),format.raw/*122.107*/(""" """),format.raw/*122.108*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="gpuMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*126.75*/i18n/*126.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*126.124*/(""":</b> <span id="y2">0</span>, <b>
                                                        """),_display_(/*127.58*/i18n/*127.62*/.getMessage("train.overview.charts.iteration")),format.raw/*127.108*/(""":</b> <span id="x2">0</span></p>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- Tables -->
                                        <div class="row-fluid">

                                                <!-- Hardware Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*139.61*/i18n/*139.65*/.getMessage("train.system.hwTable.title")),format.raw/*139.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*145.70*/i18n/*145.74*/.getMessage("train.system.hwTable.jvmCurrent")),format.raw/*145.120*/("""</th>
                                                                <th>"""),_display_(/*146.70*/i18n/*146.74*/.getMessage("train.system.hwTable.jvmMax")),format.raw/*146.116*/("""</th>
                                                                <th>"""),_display_(/*147.70*/i18n/*147.74*/.getMessage("train.system.hwTable.offHeapCurrent")),format.raw/*147.124*/("""</th>
                                                                <th>"""),_display_(/*148.70*/i18n/*148.74*/.getMessage("train.system.hwTable.offHeapMax")),format.raw/*148.120*/("""</th>
                                                                <th>"""),_display_(/*149.70*/i18n/*149.74*/.getMessage("train.system.hwTable.jvmProcs")),format.raw/*149.118*/("""</th>
                                                                <th>"""),_display_(/*150.70*/i18n/*150.74*/.getMessage("train.system.hwTable.computeDevices")),format.raw/*150.124*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="currentBytesJVM">Loading...</td>
                                                                <td id="maxBytesJVM">Loading...</td>
                                                                <td id="currentBytesOffHeap">Loading...</td>
                                                                <td id="maxBytesOffHeap">Loading...</td>
                                                                <td id="jvmAvailableProcessors">Loading...</td>
                                                                <td id="nComputeDevices">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>

                                        </div>

                                        <div class="row-fluid">

                                                <!-- Software Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*174.61*/i18n/*174.65*/.getMessage("train.system.swTable.title")),format.raw/*174.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*180.70*/i18n/*180.74*/.getMessage("train.system.swTable.hostname")),format.raw/*180.118*/("""</th>
                                                                <th>"""),_display_(/*181.70*/i18n/*181.74*/.getMessage("train.system.swTable.os")),format.raw/*181.112*/("""</th>
                                                                <th>"""),_display_(/*182.70*/i18n/*182.74*/.getMessage("train.system.swTable.osArch")),format.raw/*182.116*/("""</th>
                                                                <th>"""),_display_(/*183.70*/i18n/*183.74*/.getMessage("train.system.swTable.jvmName")),format.raw/*183.117*/("""</th>
                                                                <th>"""),_display_(/*184.70*/i18n/*184.74*/.getMessage("train.system.swTable.jvmVersion")),format.raw/*184.120*/("""</th>
                                                                <th>"""),_display_(/*185.70*/i18n/*185.74*/.getMessage("train.system.swTable.nd4jBackend")),format.raw/*185.121*/("""</th>
                                                                <th>"""),_display_(/*186.70*/i18n/*186.74*/.getMessage("train.system.swTable.nd4jDataType")),format.raw/*186.122*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="hostName">Loading...</td>
                                                                <td id="OS">Loading...</td>
                                                                <td id="OSArchitecture">Loading...</td>
                                                                <td id="jvmName">Loading...</td>
                                                                <td id="jvmVersion">Loading...</td>
                                                                <td id="nd4jBackend">Loading...</td>
                                                                <td id="nd4jDataType">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- GPU Information -->
                                        <div class="row-fluid" id="gpuTable">
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>GPU Information</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                        <!-- End System Tab -->
                                </div>
                            </div>
                        </div>
                            <!-- End System Tab -->
                    </div><!-- End Row Fluid-->
                </div><!-- End Content -->
            </div><!-- End Container-->
        </div><!-- End Row Fluid-->

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
        <script src="/assets/js/train/system.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*286.47*/("""{"""),format.raw/*286.48*/("""
                    """),format.raw/*287.21*/("""renderSystemPage();
                    renderTabs();
                    selectMachine();
                """),format.raw/*290.17*/("""}"""),format.raw/*290.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*295.41*/("""{"""),format.raw/*295.42*/("""
                    """),format.raw/*296.21*/("""renderSystemPage();
                """),format.raw/*297.17*/("""}"""),format.raw/*297.18*/(""", 2000);
        </script>
            <!--End JavaScript-->

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
object TrainingSystem extends TrainingSystem_Scope0.TrainingSystem
              /*
                  -- GENERATED --
                  DATE: Sun Nov 06 14:54:17 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: 4af7d1e2a48d45be90e43550aa8e07b3922fc313
                  MATRIX: 600->1|733->39|761->41|884->138|896->142|946->172|2731->1930|2744->1934|2795->1964|3681->2822|3695->2826|3750->2859|3900->2981|3914->2985|3966->3015|4149->3170|4163->3174|4216->3205|4364->3325|4378->3329|4434->3363|6899->5801|6912->5805|6966->5838|7774->6618|7788->6622|7856->6667|7887->6668|8263->7016|8277->7020|8345->7065|8464->7156|8478->7160|8547->7206|9023->7654|9037->7658|9102->7700|9133->7701|9506->8046|9520->8050|9588->8095|9708->8187|9722->8191|9791->8237|10424->8842|10438->8846|10502->8887|10961->9318|10975->9322|11044->9368|11148->9444|11162->9448|11227->9490|11331->9566|11345->9570|11418->9620|11522->9696|11536->9700|11605->9746|11709->9822|11723->9826|11790->9870|11894->9946|11908->9950|11981->10000|13628->11619|13642->11623|13706->11664|14165->12095|14179->12099|14246->12143|14350->12219|14364->12223|14425->12261|14529->12337|14543->12341|14608->12383|14712->12459|14726->12463|14792->12506|14896->12582|14910->12586|14979->12632|15083->12708|15097->12712|15167->12759|15271->12835|15285->12839|15356->12887|21992->19494|22022->19495|22073->19517|22212->19627|22242->19628|22410->19767|22440->19768|22491->19790|22557->19827|22587->19828
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|84->60|84->60|84->60|85->61|85->61|85->61|86->62|86->62|86->62|87->63|87->63|87->63|120->96|120->96|120->96|134->110|134->110|134->110|134->110|138->114|138->114|138->114|139->115|139->115|139->115|146->122|146->122|146->122|146->122|150->126|150->126|150->126|151->127|151->127|151->127|163->139|163->139|163->139|169->145|169->145|169->145|170->146|170->146|170->146|171->147|171->147|171->147|172->148|172->148|172->148|173->149|173->149|173->149|174->150|174->150|174->150|198->174|198->174|198->174|204->180|204->180|204->180|205->181|205->181|205->181|206->182|206->182|206->182|207->183|207->183|207->183|208->184|208->184|208->184|209->185|209->185|209->185|210->186|210->186|210->186|310->286|310->286|311->287|314->290|314->290|319->295|319->295|320->296|321->297|321->297
                  -- GENERATED --
              */
          