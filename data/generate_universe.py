import csv

blue_chips = [
    'RY.TO', 'TD.TO', 'SHOP.TO', 'CNR.TO', 'ENB.TO', 'CP.TO', 'ATD.TO', 'CNQ.TO', 'BN.TO', 'BNS.TO',
    'BMO.TO', 'TRP.TO', 'BCE.TO', 'TRI.TO', 'CSU.TO', 'ABX.TO', 'WPM.TO', 'FNV.TO', 'L.TO', 'AQN.TO',
    'T.TO', 'FTS.TO', 'GIB-A.TO', 'DOL.TO', 'MFC.TO', 'SLF.TO', 'POW.TO', 'WCN.TO', 'IMO.TO', 'SU.TO',
    'WFG.TO', 'IFC.TO', 'TECK-B.TO', 'NTR.TO', 'PPL.TO', 'CVE.TO', 'H.TO', 'RCI-B.TO', 'EMA.TO', 'GWO.TO',
    'OTEX.TO', 'GIL.TO', 'CAR-UN.TO', 'SAP.TO', 'WN.TO', 'K.TO', 'FM.TO', 'IVN.TO', 'LUN.TO', 'TOU.TO',
    'ACO-X.TO', 'AD-UN.TO', 'ALA.TO', 'ARX.TO', 'ATS.TO', 'AW-UN.TO', 'BEP-UN.TO', 'BIP-UN.TO', 'BIR.TO',
    'CASH.TO', 'CCL-B.TO', 'CF.TO', 'CFW.TO', 'CGY.TO', 'CIGI.TO', 'CJT.TO', 'CPX.TO', 'CTC-A.TO', 'CU.TO',
    'CWB.TO', 'DIR-UN.TO', 'DPM.TO', 'DRM.TO', 'ECN.TO', 'EFN.TO', 'EIF.TO', 'ELD.TO', 'EMP-A.TO', 'EQB.TO',
    'ERF.TO', 'EXE.TO', 'FFH.TO', 'FIN.TO', 'FR.TO', 'FSV.TO', 'FTT.TO', 'GEI.TO', 'GFL.TO', 'GGD.TO',
    'GUD.TO', 'HBM.TO', 'HR-UN.TO', 'IAG.TO', 'INE.TO', 'ITP.TO', 'KFS.TO', 'KEY.TO', 'KL.TO', 'KXS.TO',
    'LB.TO', 'LIF.TO', 'MAG.TO', 'MEG.TO', 'MFI.TO', 'MG.TO', 'MRE.TO', 'MSI.TO', 'MTL.TO', 'MX.TO',
    'NA.TO', 'NFI.TO', 'OGC.TO', 'ONEX.TO', 'OR.TO', 'PAAS.TO', 'PKI.TO', 'PRU.TO', 'PSI.TO', 'QSR.TO',
    'REI-UN.TO', 'RNW.TO', 'SGR-UN.TO', 'SIA.TO', 'SIS.TO', 'SJR-B.TO', 'STN.TO', 'TCL-A.TO', 'TIH.TO',
    'TOY.TO', 'TPZ.TO', 'UNV.TO', 'VET.TO', 'VII.TO', 'VNP.TO', 'WCP.TO', 'WDO.TO', 'WSP.TO', 'X.TO'
]

pennies = [
    'HIVE.TO', 'HUT.TO', 'BITF.TO', 'MN.V', 'NUMI.V', 'NILI.V', 'SLVR.V', 'DSV.V', 'LI.V', 'GIGA.V',
    'CNC.V', 'AMY.V', 'RECO.V', 'GRAT.V', 'DMGI.V', 'DGHI.V', 'GLXY.TO', 'VOXL.V', 'VPT.V', 'AIA.V',
    'EGT.V', 'FIL.TO', 'NGD.TO', 'KRR.TO', 'SOT.V', 'DOC.V', 'WELL.TO', 'CLOUD.V', 'ALEF.V', 'SKE.TO',
    'ARTG.V', 'AZM.V', 'BATT.V', 'BCN.V', 'BHS.V', 'BRC.V', 'CMMC.TO', 'CTS.TO', 'DML.TO', 'EGLX.TO',
    'ELO.V', 'EMO.V', 'EU.V', 'EXRO.TO', 'FD.V', 'FL.V', 'FOM.V', 'FTEC.V', 'GBR.V', 'GIX.V', 'GLO.V',
    'GMIN.V', 'GRSL.V', 'HITI.V', 'HPQ.V', 'ION.V', 'KNT.TO', 'LABS.TO', 'LAC.TO', 'LAAC.TO', 'LIO.V',
    'LITH.V', 'LIX.V', 'LME.V', 'MAI.V', 'MAX.V', 'MCR.V', 'MEDV.V', 'META.V', 'MOZ.V', 'MRS.V',
    'NANO.V', 'NEO.V', 'NICL.V', 'NIM.V', 'NKL.V', 'NMG.TO', 'NOU.V', 'NPRO.V', 'NSR.V', 'NWH-UN.TO',
    'NXE.TO', 'OGI.TO', 'OIII.V', 'OSI.V', 'OYL.V', 'PANO.V', 'PAT.V', 'PHOT.V', 'PMET.V', 'PRO.V',
    'PRYM.V', 'PYR.TO', 'QTRH.V', 'QUIS.V', 'RML.V', 'ROX.V', 'RSLV.V', 'SDE.TO', 'SGU.V', 'SIL.TO',
    'SOLG.TO', 'SPMT.V', 'STC.V', 'STGO.V', 'SVA.V', 'TAL.V', 'TLO.TO', 'TLRY.TO', 'TMD.V', 'TUD.V',
    'VGCX.TO', 'VGLD.V', 'VLI.V', 'VMC.V', 'VNP.V', 'VUX.V', 'WSTR.V', 'XLY.V', 'ZMA.V', 'ZON.V',
    'GSPW.V', 'HSTR.V', 'ICTV.V', 'JRV.V', 'KLY.V', 'KUL.V', 'LKY.V', 'LNC.V', 'MGRO.V', 'NDA.V',
    'NRN.V', 'NXS.V', 'OBN.V', 'OSK.TO', 'PST.V', 'QCC.V', 'RKR.V', 'SCR.TO', 'THRM.V', 'VXL.V'
]

full_list = sorted(set(blue_chips + pennies))

with open('canadian_universe.csv', 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['symbol'])
    for symbol in full_list:
        writer.writerow([symbol])

print(f"✅ Univers canadien mis à jour : {len(full_list)} tickers prêts pour le scan.")
