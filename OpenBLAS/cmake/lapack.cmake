# Sources for compiling lapack-netlib. Can't use CMakeLists.txt because lapack-netlib already has its own cmake files.

set(ALLAUX
  ilaenv.f ieeeck.f lsamen.f xerbla_array.f iparmq.f
  ilaprec.f ilatrans.f ilauplo.f iladiag.f chla_transtype.f
  ../INSTALL/ilaver.f ../INSTALL/slamch.f
)

set(SCLAUX
  sbdsdc.f
  sbdsqr.f sdisna.f slabad.f slacpy.f sladiv.f slae2.f  slaebz.f
  slaed0.f slaed1.f slaed2.f slaed3.f slaed4.f slaed5.f slaed6.f
  slaed7.f slaed8.f slaed9.f slaeda.f slaev2.f slagtf.f
  slagts.f slamrg.f slanst.f
  slapy2.f slapy3.f slarnv.f
  slarra.f slarrb.f slarrc.f slarrd.f slarre.f slarrf.f slarrj.f
  slarrk.f slarrr.f slaneg.f
  slartg.f slaruv.f slas2.f  slascl.f
  slasd0.f slasd1.f slasd2.f slasd3.f slasd4.f slasd5.f slasd6.f
  slasd7.f slasd8.f slasda.f slasdq.f slasdt.f
  slaset.f slasq1.f slasq2.f slasq3.f slasq4.f slasq5.f slasq6.f
  slasr.f  slasrt.f slassq.f slasv2.f spttrf.f sstebz.f sstedc.f
  ssteqr.f ssterf.f slaisnan.f sisnan.f
  slartgp.f slartgs.f
  ../INSTALL/second_${TIMER}.f
)

set(DZLAUX
  dbdsdc.f
  dbdsqr.f ddisna.f dlabad.f dlacpy.f dladiv.f dlae2.f  dlaebz.f
  dlaed0.f dlaed1.f dlaed2.f dlaed3.f dlaed4.f dlaed5.f dlaed6.f
  dlaed7.f dlaed8.f dlaed9.f dlaeda.f dlaev2.f dlagtf.f
  dlagts.f dlamrg.f dlanst.f
  dlapy2.f dlapy3.f dlarnv.f
  dlarra.f dlarrb.f dlarrc.f dlarrd.f dlarre.f dlarrf.f dlarrj.f
  dlarrk.f dlarrr.f dlaneg.f
  dlartg.f dlaruv.f dlas2.f  dlascl.f
  dlasd0.f dlasd1.f dlasd2.f dlasd3.f dlasd4.f dlasd5.f dlasd6.f
  dlasd7.f dlasd8.f dlasda.f dlasdq.f dlasdt.f
  dlaset.f dlasq1.f dlasq2.f dlasq3.f dlasq4.f dlasq5.f dlasq6.f
  dlasr.f  dlasrt.f dlassq.f dlasv2.f dpttrf.f dstebz.f dstedc.f
  dsteqr.f dsterf.f dlaisnan.f disnan.f
  dlartgp.f dlartgs.f
  ../INSTALL/dlamch.f ../INSTALL/dsecnd_${TIMER}.f
)

set(SLASRC
  sgbbrd.f sgbcon.f sgbequ.f sgbrfs.f sgbsv.f
  sgbsvx.f sgbtf2.f sgbtrf.f sgbtrs.f sgebak.f sgebal.f sgebd2.f
  sgebrd.f sgecon.f sgeequ.f sgees.f  sgeesx.f sgeev.f  sgeevx.f
  DEPRECATED/sgegs.f  DEPRECATED/sgegv.f  sgehd2.f sgehrd.f sgelq2.f sgelqf.f
  sgels.f  sgelsd.f sgelss.f DEPRECATED/sgelsx.f sgelsy.f sgeql2.f sgeqlf.f
  sgeqp3.f DEPRECATED/sgeqpf.f sgeqr2.f sgeqr2p.f sgeqrf.f sgeqrfp.f sgerfs.f
  sgerq2.f sgerqf.f sgesc2.f sgesdd.f  sgesvd.f sgesvx.f
  sgetc2.f sgetri.f
  sggbak.f sggbal.f sgges.f  sggesx.f sggev.f  sggevx.f
  sggglm.f sgghrd.f sgglse.f sggqrf.f
  sggrqf.f DEPRECATED/sggsvd.f DEPRECATED/sggsvp.f sgtcon.f sgtrfs.f sgtsv.f
  sgtsvx.f sgttrf.f sgttrs.f sgtts2.f shgeqz.f
  shsein.f shseqr.f slabrd.f slacon.f slacn2.f
  slaein.f slaexc.f slag2.f  slags2.f slagtm.f slagv2.f slahqr.f
  DEPRECATED/slahrd.f slahr2.f slaic1.f slaln2.f slals0.f slalsa.f slalsd.f
  slangb.f slange.f slangt.f slanhs.f slansb.f slansp.f
  slansy.f slantb.f slantp.f slantr.f slanv2.f
  slapll.f slapmt.f
  slaqgb.f slaqge.f slaqp2.f slaqps.f slaqsb.f slaqsp.f slaqsy.f
  slaqr0.f slaqr1.f slaqr2.f slaqr3.f slaqr4.f slaqr5.f
  slaqtr.f slar1v.f slar2v.f ilaslr.f ilaslc.f
  slarf.f  slarfb.f slarfg.f slarfgp.f slarft.f slarfx.f slargv.f
  slarrv.f slartv.f
  slarz.f  slarzb.f slarzt.f slasy2.f slasyf.f slasyf_rook.f
  slatbs.f slatdf.f slatps.f slatrd.f slatrs.f slatrz.f DEPRECATED/slatzm.f
  sopgtr.f sopmtr.f sorg2l.f sorg2r.f
  sorgbr.f sorghr.f sorgl2.f sorglq.f sorgql.f sorgqr.f sorgr2.f
  sorgrq.f sorgtr.f sorm2l.f sorm2r.f
  sormbr.f sormhr.f sorml2.f sormlq.f sormql.f sormqr.f sormr2.f
  sormr3.f sormrq.f sormrz.f sormtr.f spbcon.f spbequ.f spbrfs.f
  spbstf.f spbsv.f  spbsvx.f
  spbtf2.f spbtrf.f spbtrs.f spocon.f spoequ.f sporfs.f sposv.f
  sposvx.f spstrf.f spstf2.f
  sppcon.f sppequ.f
  spprfs.f sppsv.f  sppsvx.f spptrf.f spptri.f spptrs.f sptcon.f
  spteqr.f sptrfs.f sptsv.f  sptsvx.f spttrs.f sptts2.f srscl.f
  ssbev.f  ssbevd.f ssbevx.f ssbgst.f ssbgv.f  ssbgvd.f ssbgvx.f
  ssbtrd.f sspcon.f sspev.f  sspevd.f sspevx.f sspgst.f
  sspgv.f  sspgvd.f sspgvx.f ssprfs.f sspsv.f  sspsvx.f ssptrd.f
  ssptrf.f ssptri.f ssptrs.f sstegr.f sstein.f sstev.f  sstevd.f sstevr.f
  sstevx.f
  ssycon.f ssyev.f  ssyevd.f ssyevr.f ssyevx.f ssygs2.f
  ssygst.f ssygv.f  ssygvd.f ssygvx.f ssyrfs.f ssysv.f  ssysvx.f
  ssytd2.f ssytf2.f ssytrd.f ssytrf.f ssytri.f ssytri2.f ssytri2x.f
  ssyswapr.f ssytrs.f ssytrs2.f ssyconv.f
  ssytf2_rook.f ssytrf_rook.f ssytrs_rook.f
  ssytri_rook.f ssycon_rook.f ssysv_rook.f
  stbcon.f
  stbrfs.f stbtrs.f stgevc.f stgex2.f stgexc.f stgsen.f
  stgsja.f stgsna.f stgsy2.f stgsyl.f stpcon.f stprfs.f stptri.f
  stptrs.f
  strcon.f strevc.f strexc.f strrfs.f strsen.f strsna.f strsyl.f
  strtrs.f DEPRECATED/stzrqf.f stzrzf.f sstemr.f
  slansf.f spftrf.f spftri.f spftrs.f ssfrk.f stfsm.f stftri.f stfttp.f
  stfttr.f stpttf.f stpttr.f strttf.f strttp.f
  sgejsv.f  sgesvj.f  sgsvj0.f  sgsvj1.f
  sgeequb.f ssyequb.f spoequb.f sgbequb.f
  sbbcsd.f slapmr.f sorbdb.f sorbdb1.f sorbdb2.f sorbdb3.f sorbdb4.f
  sorbdb5.f sorbdb6.f sorcsd.f sorcsd2by1.f
  sgeqrt.f sgeqrt2.f sgeqrt3.f sgemqrt.f
  stpqrt.f stpqrt2.f stpmqrt.f stprfb.f spotri.f
)

set(DSLASRC spotrs.f)

set(CLASRC
  cbdsqr.f cgbbrd.f cgbcon.f cgbequ.f cgbrfs.f cgbsv.f  cgbsvx.f
  cgbtf2.f cgbtrf.f cgbtrs.f cgebak.f cgebal.f cgebd2.f cgebrd.f
  cgecon.f cgeequ.f cgees.f  cgeesx.f cgeev.f  cgeevx.f
  DEPRECATED/cgegs.f  DEPRECATED/cgegv.f  cgehd2.f cgehrd.f cgelq2.f cgelqf.f
  cgels.f  cgelsd.f cgelss.f DEPRECATED/cgelsx.f cgelsy.f cgeql2.f cgeqlf.f cgeqp3.f
  DEPRECATED/cgeqpf.f cgeqr2.f cgeqr2p.f cgeqrf.f cgeqrfp.f cgerfs.f
  cgerq2.f cgerqf.f cgesc2.f cgesdd.f  cgesvd.f
  cgesvx.f cgetc2.f cgetri.f
  cggbak.f cggbal.f cgges.f  cggesx.f cggev.f  cggevx.f cggglm.f
  cgghrd.f cgglse.f cggqrf.f cggrqf.f
  DEPRECATED/cggsvd.f DEPRECATED/cggsvp.f
  cgtcon.f cgtrfs.f cgtsv.f  cgtsvx.f cgttrf.f cgttrs.f cgtts2.f chbev.f
  chbevd.f chbevx.f chbgst.f chbgv.f  chbgvd.f chbgvx.f chbtrd.f
  checon.f cheev.f  cheevd.f cheevr.f cheevx.f chegs2.f chegst.f
  chegv.f  chegvd.f chegvx.f cherfs.f chesv.f  chesvx.f chetd2.f
  chetf2.f chetrd.f
  chetrf.f chetri.f chetri2.f chetri2x.f cheswapr.f
  chetrs.f chetrs2.f
  chetf2_rook.f chetrf_rook.f chetri_rook.f chetrs_rook.f checon_rook.f chesv_rook.f
  chgeqz.f chpcon.f chpev.f  chpevd.f
  chpevx.f chpgst.f chpgv.f  chpgvd.f chpgvx.f chprfs.f chpsv.f
  chpsvx.f
  chptrd.f chptrf.f chptri.f chptrs.f chsein.f chseqr.f clabrd.f
  clacgv.f clacon.f clacn2.f clacp2.f clacpy.f clacrm.f clacrt.f cladiv.f
  claed0.f claed7.f claed8.f
  claein.f claesy.f claev2.f clags2.f clagtm.f
  clahef.f clahef_rook.f clahqr.f
  DEPRECATED/clahrd.f clahr2.f claic1.f clals0.f clalsa.f clalsd.f clangb.f clange.f clangt.f
  clanhb.f clanhe.f
  clanhp.f clanhs.f clanht.f clansb.f clansp.f clansy.f clantb.f
  clantp.f clantr.f clapll.f clapmt.f clarcm.f claqgb.f claqge.f
  claqhb.f claqhe.f claqhp.f claqp2.f claqps.f claqsb.f
  claqr0.f claqr1.f claqr2.f claqr3.f claqr4.f claqr5.f
  claqsp.f claqsy.f clar1v.f clar2v.f ilaclr.f ilaclc.f
  clarf.f  clarfb.f clarfg.f clarft.f clarfgp.f
  clarfx.f clargv.f clarnv.f clarrv.f clartg.f clartv.f
  clarz.f  clarzb.f clarzt.f clascl.f claset.f clasr.f  classq.f
  clasyf.f clasyf_rook.f clatbs.f clatdf.f clatps.f clatrd.f clatrs.f clatrz.f
  DEPRECATED/clatzm.f cpbcon.f cpbequ.f cpbrfs.f cpbstf.f cpbsv.f
  cpbsvx.f cpbtf2.f cpbtrf.f cpbtrs.f cpocon.f cpoequ.f cporfs.f
  cposv.f  cposvx.f cpstrf.f cpstf2.f
  cppcon.f cppequ.f cpprfs.f cppsv.f  cppsvx.f cpptrf.f cpptri.f cpptrs.f
  cptcon.f cpteqr.f cptrfs.f cptsv.f  cptsvx.f cpttrf.f cpttrs.f cptts2.f
  crot.f   cspcon.f csprfs.f cspsv.f
  cspsvx.f csptrf.f csptri.f csptrs.f csrscl.f cstedc.f
  cstegr.f cstein.f csteqr.f
  csycon.f
  csyrfs.f csysv.f  csysvx.f csytf2.f csytrf.f csytri.f csytri2.f csytri2x.f
  csyswapr.f csytrs.f csytrs2.f csyconv.f
  csytf2_rook.f csytrf_rook.f csytrs_rook.f
  csytri_rook.f csycon_rook.f csysv_rook.f
  ctbcon.f ctbrfs.f ctbtrs.f ctgevc.f ctgex2.f
  ctgexc.f ctgsen.f ctgsja.f ctgsna.f ctgsy2.f ctgsyl.f ctpcon.f
  ctprfs.f ctptri.f
  ctptrs.f ctrcon.f ctrevc.f ctrexc.f ctrrfs.f ctrsen.f ctrsna.f
  ctrsyl.f ctrtrs.f DEPRECATED/ctzrqf.f ctzrzf.f cung2l.f cung2r.f
  cungbr.f cunghr.f cungl2.f cunglq.f cungql.f cungqr.f cungr2.f
  cungrq.f cungtr.f cunm2l.f cunm2r.f cunmbr.f cunmhr.f cunml2.f
  cunmlq.f cunmql.f cunmqr.f cunmr2.f cunmr3.f cunmrq.f cunmrz.f
  cunmtr.f cupgtr.f cupmtr.f icmax1.f scsum1.f cstemr.f
  chfrk.f ctfttp.f clanhf.f cpftrf.f cpftri.f cpftrs.f ctfsm.f ctftri.f
  ctfttr.f ctpttf.f ctpttr.f ctrttf.f ctrttp.f
  cgeequb.f cgbequb.f csyequb.f cpoequb.f cheequb.f
  cbbcsd.f clapmr.f cunbdb.f cunbdb1.f cunbdb2.f cunbdb3.f cunbdb4.f
  cunbdb5.f cunbdb6.f cuncsd.f cuncsd2by1.f
  cgeqrt.f cgeqrt2.f cgeqrt3.f cgemqrt.f
  ctpqrt.f ctpqrt2.f ctpmqrt.f ctprfb.f cpotri.f
)

set(ZCLASRC cpotrs.f)

set(DLASRC
  dgbbrd.f dgbcon.f dgbequ.f dgbrfs.f dgbsv.f
  dgbsvx.f dgbtf2.f dgbtrf.f dgbtrs.f dgebak.f dgebal.f dgebd2.f
  dgebrd.f dgecon.f dgeequ.f dgees.f  dgeesx.f dgeev.f  dgeevx.f
  DEPRECATED/dgegs.f  DEPRECATED/dgegv.f  dgehd2.f dgehrd.f dgelq2.f dgelqf.f
  dgels.f  dgelsd.f dgelss.f DEPRECATED/dgelsx.f dgelsy.f dgeql2.f dgeqlf.f
  dgeqp3.f DEPRECATED/dgeqpf.f dgeqr2.f dgeqr2p.f dgeqrf.f dgeqrfp.f dgerfs.f
  dgerq2.f dgerqf.f dgesc2.f dgesdd.f  dgesvd.f dgesvx.f
  dgetc2.f dgetri.f
  dggbak.f dggbal.f dgges.f  dggesx.f dggev.f  dggevx.f
  dggglm.f dgghrd.f dgglse.f dggqrf.f
  dggrqf.f DEPRECATED/dggsvd.f DEPRECATED/dggsvp.f dgtcon.f dgtrfs.f dgtsv.f
  dgtsvx.f dgttrf.f dgttrs.f dgtts2.f dhgeqz.f
  dhsein.f dhseqr.f dlabrd.f dlacon.f dlacn2.f
  dlaein.f dlaexc.f dlag2.f  dlags2.f dlagtm.f dlagv2.f dlahqr.f
  DEPRECATED/dlahrd.f dlahr2.f dlaic1.f dlaln2.f dlals0.f dlalsa.f dlalsd.f
  dlangb.f dlange.f dlangt.f dlanhs.f dlansb.f dlansp.f
  dlansy.f dlantb.f dlantp.f dlantr.f dlanv2.f
  dlapll.f dlapmt.f
  dlaqgb.f dlaqge.f dlaqp2.f dlaqps.f dlaqsb.f dlaqsp.f dlaqsy.f
  dlaqr0.f dlaqr1.f dlaqr2.f dlaqr3.f dlaqr4.f dlaqr5.f
  dlaqtr.f dlar1v.f dlar2v.f iladlr.f iladlc.f
  dlarf.f  dlarfb.f dlarfg.f dlarfgp.f dlarft.f dlarfx.f
  dlargv.f dlarrv.f dlartv.f
  dlarz.f  dlarzb.f dlarzt.f dlasy2.f dlasyf.f dlasyf_rook.f
  dlatbs.f dlatdf.f dlatps.f dlatrd.f dlatrs.f dlatrz.f DEPRECATED/dlatzm.f
  dopgtr.f dopmtr.f dorg2l.f dorg2r.f
  dorgbr.f dorghr.f dorgl2.f dorglq.f dorgql.f dorgqr.f dorgr2.f
  dorgrq.f dorgtr.f dorm2l.f dorm2r.f
  dormbr.f dormhr.f dorml2.f dormlq.f dormql.f dormqr.f dormr2.f
  dormr3.f dormrq.f dormrz.f dormtr.f dpbcon.f dpbequ.f dpbrfs.f
  dpbstf.f dpbsv.f  dpbsvx.f
  dpbtf2.f dpbtrf.f dpbtrs.f dpocon.f dpoequ.f dporfs.f dposv.f
  dposvx.f dpotrs.f dpstrf.f dpstf2.f
  dppcon.f dppequ.f
  dpprfs.f dppsv.f  dppsvx.f dpptrf.f dpptri.f dpptrs.f dptcon.f
  dpteqr.f dptrfs.f dptsv.f  dptsvx.f dpttrs.f dptts2.f drscl.f
  dsbev.f  dsbevd.f dsbevx.f dsbgst.f dsbgv.f  dsbgvd.f dsbgvx.f
  dsbtrd.f  dspcon.f dspev.f  dspevd.f dspevx.f dspgst.f
  dspgv.f  dspgvd.f dspgvx.f dsprfs.f dspsv.f  dspsvx.f dsptrd.f
  dsptrf.f dsptri.f dsptrs.f dstegr.f dstein.f dstev.f  dstevd.f dstevr.f
  dstevx.f
  dsycon.f dsyev.f  dsyevd.f dsyevr.f
  dsyevx.f dsygs2.f dsygst.f dsygv.f  dsygvd.f dsygvx.f dsyrfs.f
  dsysv.f  dsysvx.f
  dsytd2.f dsytf2.f dsytrd.f dsytrf.f dsytri.f dsytri2.f dsytri2x.f
  dsyswapr.f dsytrs.f dsytrs2.f dsyconv.f
  dsytf2_rook.f dsytrf_rook.f dsytrs_rook.f
  dsytri_rook.f dsycon_rook.f dsysv_rook.f
  dtbcon.f dtbrfs.f dtbtrs.f dtgevc.f dtgex2.f dtgexc.f dtgsen.f
  dtgsja.f dtgsna.f dtgsy2.f dtgsyl.f dtpcon.f dtprfs.f dtptri.f
  dtptrs.f
  dtrcon.f dtrevc.f dtrexc.f dtrrfs.f dtrsen.f dtrsna.f dtrsyl.f
  dtrtrs.f DEPRECATED/dtzrqf.f dtzrzf.f dstemr.f
  dsgesv.f dsposv.f dlag2s.f slag2d.f dlat2s.f
  dlansf.f dpftrf.f dpftri.f dpftrs.f dsfrk.f dtfsm.f dtftri.f dtfttp.f
  dtfttr.f dtpttf.f dtpttr.f dtrttf.f dtrttp.f
  dgejsv.f  dgesvj.f  dgsvj0.f  dgsvj1.f
  dgeequb.f dsyequb.f dpoequb.f dgbequb.f
  dbbcsd.f dlapmr.f dorbdb.f dorbdb1.f dorbdb2.f dorbdb3.f dorbdb4.f
  dorbdb5.f dorbdb6.f dorcsd.f dorcsd2by1.f
  dgeqrt.f dgeqrt2.f dgeqrt3.f dgemqrt.f
  dtpqrt.f dtpqrt2.f dtpmqrt.f dtprfb.f dpotri.f
)

set(ZLASRC
  zbdsqr.f zgbbrd.f zgbcon.f zgbequ.f zgbrfs.f zgbsv.f  zgbsvx.f
  zgbtf2.f zgbtrf.f zgbtrs.f zgebak.f zgebal.f zgebd2.f zgebrd.f
  zgecon.f zgeequ.f zgees.f  zgeesx.f zgeev.f  zgeevx.f
  DEPRECATED/zgegs.f  DEPRECATED/zgegv.f  zgehd2.f zgehrd.f zgelq2.f zgelqf.f
  zgels.f  zgelsd.f zgelss.f DEPRECATED/zgelsx.f zgelsy.f zgeql2.f zgeqlf.f zgeqp3.f
  DEPRECATED/zgeqpf.f zgeqr2.f zgeqr2p.f zgeqrf.f zgeqrfp.f zgerfs.f zgerq2.f zgerqf.f
  zgesc2.f zgesdd.f zgesvd.f zgesvx.f zgetc2.f
  zgetri.f
  zggbak.f zggbal.f zgges.f  zggesx.f zggev.f  zggevx.f zggglm.f
  zgghrd.f zgglse.f zggqrf.f zggrqf.f
  DEPRECATED/zggsvd.f DEPRECATED/zggsvp.f
  zgtcon.f zgtrfs.f zgtsv.f  zgtsvx.f zgttrf.f zgttrs.f zgtts2.f zhbev.f
  zhbevd.f zhbevx.f zhbgst.f zhbgv.f  zhbgvd.f zhbgvx.f zhbtrd.f
  zhecon.f zheev.f  zheevd.f zheevr.f zheevx.f zhegs2.f zhegst.f
  zhegv.f  zhegvd.f zhegvx.f zherfs.f zhesv.f  zhesvx.f zhetd2.f
  zhetf2.f zhetrd.f
  zhetrf.f zhetri.f zhetri2.f zhetri2x.f zheswapr.f
  zhetrs.f zhetrs2.f
  zhetf2_rook.f zhetrf_rook.f zhetri_rook.f zhetrs_rook.f zhecon_rook.f zhesv_rook.f
  zhgeqz.f zhpcon.f zhpev.f  zhpevd.f
  zhpevx.f zhpgst.f zhpgv.f  zhpgvd.f zhpgvx.f zhprfs.f zhpsv.f
  zhpsvx.f
  zhptrd.f zhptrf.f zhptri.f zhptrs.f zhsein.f zhseqr.f zlabrd.f
  zlacgv.f zlacon.f zlacn2.f zlacp2.f zlacpy.f zlacrm.f zlacrt.f zladiv.f
  zlaed0.f zlaed7.f zlaed8.f
  zlaein.f zlaesy.f zlaev2.f zlags2.f zlagtm.f
  zlahef.f zlahef_rook.f zlahqr.f
  DEPRECATED/zlahrd.f zlahr2.f zlaic1.f zlals0.f zlalsa.f zlalsd.f zlangb.f zlange.f
  zlangt.f zlanhb.f
  zlanhe.f
  zlanhp.f zlanhs.f zlanht.f zlansb.f zlansp.f zlansy.f zlantb.f
  zlantp.f zlantr.f zlapll.f zlapmt.f zlaqgb.f zlaqge.f
  zlaqhb.f zlaqhe.f zlaqhp.f zlaqp2.f zlaqps.f zlaqsb.f
  zlaqr0.f zlaqr1.f zlaqr2.f zlaqr3.f zlaqr4.f zlaqr5.f
  zlaqsp.f zlaqsy.f zlar1v.f zlar2v.f ilazlr.f ilazlc.f
  zlarcm.f zlarf.f  zlarfb.f
  zlarfg.f zlarft.f zlarfgp.f
  zlarfx.f zlargv.f zlarnv.f zlarrv.f zlartg.f zlartv.f
  zlarz.f  zlarzb.f zlarzt.f zlascl.f zlaset.f zlasr.f
  zlassq.f zlasyf.f zlasyf_rook.f
  zlatbs.f zlatdf.f zlatps.f zlatrd.f zlatrs.f zlatrz.f DEPRECATED/zlatzm.f
  zpbcon.f zpbequ.f zpbrfs.f zpbstf.f zpbsv.f
  zpbsvx.f zpbtf2.f zpbtrf.f zpbtrs.f zpocon.f zpoequ.f zporfs.f
  zposv.f  zposvx.f zpotrs.f zpstrf.f zpstf2.f
  zppcon.f zppequ.f zpprfs.f zppsv.f  zppsvx.f zpptrf.f zpptri.f zpptrs.f
  zptcon.f zpteqr.f zptrfs.f zptsv.f  zptsvx.f zpttrf.f zpttrs.f zptts2.f
  zrot.f   zspcon.f zsprfs.f zspsv.f
  zspsvx.f zsptrf.f zsptri.f zsptrs.f zdrscl.f zstedc.f
  zstegr.f zstein.f zsteqr.f
  zsycon.f
  zsyrfs.f zsysv.f  zsysvx.f zsytf2.f zsytrf.f zsytri.f zsytri2.f zsytri2x.f
  zsyswapr.f zsytrs.f zsytrs2.f zsyconv.f
  zsytf2_rook.f zsytrf_rook.f zsytrs_rook.f
  zsytri_rook.f zsycon_rook.f zsysv_rook.f
  ztbcon.f ztbrfs.f ztbtrs.f ztgevc.f ztgex2.f
  ztgexc.f ztgsen.f ztgsja.f ztgsna.f ztgsy2.f ztgsyl.f ztpcon.f
  ztprfs.f ztptri.f
  ztptrs.f ztrcon.f ztrevc.f ztrexc.f ztrrfs.f ztrsen.f ztrsna.f
  ztrsyl.f ztrtrs.f DEPRECATED/ztzrqf.f ztzrzf.f zung2l.f
  zung2r.f zungbr.f zunghr.f zungl2.f zunglq.f zungql.f zungqr.f zungr2.f
  zungrq.f zungtr.f zunm2l.f zunm2r.f zunmbr.f zunmhr.f zunml2.f
  zunmlq.f zunmql.f zunmqr.f zunmr2.f zunmr3.f zunmrq.f zunmrz.f
  zunmtr.f zupgtr.f
  zupmtr.f izmax1.f dzsum1.f zstemr.f
  zcgesv.f zcposv.f zlag2c.f clag2z.f zlat2c.f
  zhfrk.f ztfttp.f zlanhf.f zpftrf.f zpftri.f zpftrs.f ztfsm.f ztftri.f
  ztfttr.f ztpttf.f ztpttr.f ztrttf.f ztrttp.f
  zgeequb.f zgbequb.f zsyequb.f zpoequb.f zheequb.f
  zbbcsd.f zlapmr.f zunbdb.f zunbdb1.f zunbdb2.f zunbdb3.f zunbdb4.f
  zunbdb5.f zunbdb6.f zuncsd.f zuncsd2by1.f
  zgeqrt.f zgeqrt2.f zgeqrt3.f zgemqrt.f
  ztpqrt.f ztpqrt2.f ztpmqrt.f ztprfb.f zpotri.f
)

set(LA_REL_SRC ${ALLAUX})
if (BUILD_SINGLE)
  list(APPEND LA_REL_SRC ${SLASRC} ${DSLASRC} ${SCLAUX})
endif ()

if (BUILD_DOUBLE)
  list(APPEND LA_REL_SRC ${DLASRC} ${DSLASRC} ${DZLAUX})
endif ()

if (BUILD_COMPLEX)
  list(APPEND LA_REL_SRC ${CLASRC} ${ZCLASRC} ${SCLAUX})
endif ()

if (BUILD_COMPLEX16)
  list(APPEND LA_REL_SRC ${ZLASRC} ${ZCLASRC} ${DZLAUX})
endif ()

# add lapack-netlib folder to the sources
set(LA_SOURCES "")
foreach (LA_FILE ${LA_REL_SRC})
  list(APPEND LA_SOURCES "${NETLIB_LAPACK_DIR}/SRC/${LA_FILE}")
endforeach ()
set_source_files_properties(${LA_SOURCES} PROPERTIES COMPILE_FLAGS "${LAPACK_FFLAGS}")
