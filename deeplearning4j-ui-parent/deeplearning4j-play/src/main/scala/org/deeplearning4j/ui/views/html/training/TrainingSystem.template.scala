
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
                        """),_display_(/*43.26*/i18n/*43.30*/.getMessage("train.session.label")),format.raw/*43.64*/("""
                        """),format.raw/*44.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
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
                            """),format.raw/*63.161*/("""
                            """),format.raw/*64.29*/("""<li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">
                                    """),_display_(/*66.38*/i18n/*66.42*/.getMessage("train.nav.language")),format.raw/*66.75*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('de', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> Deutsch</span></a></li>
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
                                <h2><b>"""),_display_(/*97.41*/i18n/*97.45*/.getMessage("train.system.title")),format.raw/*97.78*/("""</b></h2>
                                <div class="btn-group" style="margin-top: -11px; position:absolute; right: 40px;">
                                <button class="btn dropdown-toggle btn-primary" data-toggle="dropdown">"""),_display_(/*99.105*/i18n/*99.109*/.getMessage("train.system.selectMachine")),format.raw/*99.150*/(""" """),format.raw/*99.151*/("""<span class="caret"></span></button>
                                    <ul class="dropdown-menu" id="systemTab"></ul>
                                </div>
                            </div>
                            <div class="box-content">

                                    <!--Start System Information -->
                                <div class="tab-content">
                                    <div class="tab-pane active">

                                            <!-- System Memory Utilization Chart -->
                                        <div class="row-fluid">

                                            <div class="box span12" id="systemMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*114.61*/i18n/*114.65*/.getMessage("train.system.chart.systemMemoryTitle")),format.raw/*114.116*/(""" """),format.raw/*114.117*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="systemMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*118.75*/i18n/*118.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*118.124*/(""":</b> <span id="y">0</span>, <b>
                                                        """),_display_(/*119.58*/i18n/*119.62*/.getMessage("train.overview.charts.iteration")),format.raw/*119.108*/(""":</b> <span id="x">0</span></p>
                                                </div>
                                            </div>

                                            <!-- GPU Memory Utlization Chart -->
                                            <div class="box span6" id="gpuMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*126.61*/i18n/*126.65*/.getMessage("train.system.chart.gpuMemoryTitle")),format.raw/*126.113*/(""" """),format.raw/*126.114*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="gpuMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*130.75*/i18n/*130.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*130.124*/(""":</b> <span id="y2">0</span>, <b>
                                                        """),_display_(/*131.58*/i18n/*131.62*/.getMessage("train.overview.charts.iteration")),format.raw/*131.108*/(""":</b> <span id="x2">0</span></p>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- Tables -->
                                        <div class="row-fluid">

                                                <!-- Hardware Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*143.61*/i18n/*143.65*/.getMessage("train.system.hwTable.title")),format.raw/*143.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*149.70*/i18n/*149.74*/.getMessage("train.system.hwTable.jvmCurrent")),format.raw/*149.120*/("""</th>
                                                                <th>"""),_display_(/*150.70*/i18n/*150.74*/.getMessage("train.system.hwTable.jvmMax")),format.raw/*150.116*/("""</th>
                                                                <th>"""),_display_(/*151.70*/i18n/*151.74*/.getMessage("train.system.hwTable.offHeapCurrent")),format.raw/*151.124*/("""</th>
                                                                <th>"""),_display_(/*152.70*/i18n/*152.74*/.getMessage("train.system.hwTable.offHeapMax")),format.raw/*152.120*/("""</th>
                                                                <th>"""),_display_(/*153.70*/i18n/*153.74*/.getMessage("train.system.hwTable.jvmProcs")),format.raw/*153.118*/("""</th>
                                                                <th>"""),_display_(/*154.70*/i18n/*154.74*/.getMessage("train.system.hwTable.computeDevices")),format.raw/*154.124*/("""</th>
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
                                                    <h2><b>"""),_display_(/*178.61*/i18n/*178.65*/.getMessage("train.system.swTable.title")),format.raw/*178.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*184.70*/i18n/*184.74*/.getMessage("train.system.swTable.hostname")),format.raw/*184.118*/("""</th>
                                                                <th>"""),_display_(/*185.70*/i18n/*185.74*/.getMessage("train.system.swTable.os")),format.raw/*185.112*/("""</th>
                                                                <th>"""),_display_(/*186.70*/i18n/*186.74*/.getMessage("train.system.swTable.osArch")),format.raw/*186.116*/("""</th>
                                                                <th>"""),_display_(/*187.70*/i18n/*187.74*/.getMessage("train.system.swTable.jvmName")),format.raw/*187.117*/("""</th>
                                                                <th>"""),_display_(/*188.70*/i18n/*188.74*/.getMessage("train.system.swTable.jvmVersion")),format.raw/*188.120*/("""</th>
                                                                <th>"""),_display_(/*189.70*/i18n/*189.74*/.getMessage("train.system.swTable.nd4jBackend")),format.raw/*189.121*/("""</th>
                                                                <th>"""),_display_(/*190.70*/i18n/*190.74*/.getMessage("train.system.swTable.nd4jDataType")),format.raw/*190.122*/("""</th>
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

                                            """),format.raw/*210.73*/("""
                                        """),format.raw/*211.82*/("""
                                            """),format.raw/*212.73*/("""
                                                """),format.raw/*213.77*/("""
                                                    """),format.raw/*214.88*/("""
                                                """),format.raw/*215.59*/("""
                                                """),format.raw/*216.78*/("""
                                                    """),format.raw/*217.92*/("""
                                                        """),format.raw/*218.68*/("""
                                                            """),format.raw/*219.69*/("""
                                                                """),format.raw/*220.79*/("""
                                                            """),format.raw/*221.70*/("""
                                                        """),format.raw/*222.69*/("""
                                                        """),format.raw/*223.68*/("""
                                                            """),format.raw/*224.69*/("""
                                                                """),format.raw/*225.108*/("""
                                                            """),format.raw/*226.70*/("""
                                                        """),format.raw/*227.69*/("""
                                                    """),format.raw/*228.65*/("""
                                                """),format.raw/*229.59*/("""
                                            """),format.raw/*230.55*/("""
                                        """),format.raw/*231.51*/("""
                                    """),format.raw/*232.37*/("""</div>
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
                $(document).ready(function () """),format.raw/*278.47*/("""{"""),format.raw/*278.48*/("""
                    """),format.raw/*279.21*/("""renderSystemPage(true);
                    renderTabs();
                    selectMachine();
                    /* Default GPU to hidden */
                    $("#gpuTable").hide();
                    $("#gpuMemoryChart").hide();
                """),format.raw/*285.17*/("""}"""),format.raw/*285.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*290.41*/("""{"""),format.raw/*290.42*/("""
                    """),format.raw/*291.21*/("""renderSystemPage(false);
                """),format.raw/*292.17*/("""}"""),format.raw/*292.18*/(""", 2000);
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
                  DATE: Sun Jan 08 12:25:15 AEDT 2017
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: f6dc816c7d21c4fb3ec281a7215a95ae05e8c217
                  MATRIX: 600->1|733->39|761->41|884->138|896->142|946->172|2731->1930|2744->1934|2795->1964|2943->2085|2956->2089|3011->2123|3065->2149|3797->2853|3811->2857|3866->2890|4016->3012|4030->3016|4082->3046|4265->3201|4279->3205|4332->3236|4407->3414|4465->3444|4681->3633|4694->3637|4748->3670|7182->6077|7195->6081|7249->6114|7508->6345|7522->6349|7585->6390|7615->6391|8476->7224|8490->7228|8564->7279|8595->7280|8971->7628|8985->7632|9053->7677|9172->7768|9186->7772|9255->7818|9731->8266|9745->8270|9816->8318|9847->8319|10220->8664|10234->8668|10302->8713|10422->8805|10436->8809|10505->8855|11138->9460|11152->9464|11216->9505|11675->9936|11689->9940|11758->9986|11862->10062|11876->10066|11941->10108|12045->10184|12059->10188|12132->10238|12236->10314|12250->10318|12319->10364|12423->10440|12437->10444|12504->10488|12608->10564|12622->10568|12695->10618|14342->12237|14356->12241|14420->12282|14879->12713|14893->12717|14960->12761|15064->12837|15078->12841|15139->12879|15243->12955|15257->12959|15322->13001|15426->13077|15440->13081|15506->13124|15610->13200|15624->13204|15693->13250|15797->13326|15811->13330|15881->13377|15985->13453|15999->13457|16070->13505|17470->14904|17541->14987|17616->15061|17695->15139|17778->15228|17857->15288|17936->15367|18019->15460|18106->15529|18197->15599|18292->15679|18383->15750|18470->15820|18557->15889|18648->15959|18744->16068|18835->16139|18922->16209|19005->16275|19084->16335|19159->16391|19230->16443|19297->16481|21959->19114|21989->19115|22040->19137|22326->19394|22356->19395|22524->19534|22554->19535|22605->19557|22676->19599|22706->19600
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|67->43|67->43|67->43|68->44|84->60|84->60|84->60|85->61|85->61|85->61|86->62|86->62|86->62|87->63|88->64|90->66|90->66|90->66|121->97|121->97|121->97|123->99|123->99|123->99|123->99|138->114|138->114|138->114|138->114|142->118|142->118|142->118|143->119|143->119|143->119|150->126|150->126|150->126|150->126|154->130|154->130|154->130|155->131|155->131|155->131|167->143|167->143|167->143|173->149|173->149|173->149|174->150|174->150|174->150|175->151|175->151|175->151|176->152|176->152|176->152|177->153|177->153|177->153|178->154|178->154|178->154|202->178|202->178|202->178|208->184|208->184|208->184|209->185|209->185|209->185|210->186|210->186|210->186|211->187|211->187|211->187|212->188|212->188|212->188|213->189|213->189|213->189|214->190|214->190|214->190|234->210|235->211|236->212|237->213|238->214|239->215|240->216|241->217|242->218|243->219|244->220|245->221|246->222|247->223|248->224|249->225|250->226|251->227|252->228|253->229|254->230|255->231|256->232|302->278|302->278|303->279|309->285|309->285|314->290|314->290|315->291|316->292|316->292
                  -- GENERATED --
              */
          