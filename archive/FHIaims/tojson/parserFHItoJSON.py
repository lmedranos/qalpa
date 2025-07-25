from sys  import argv
from os.path import isfile, join
from os import getcwd
import re
import json
import numpy as np

def get_data(log):
    lines = open(log, 'r').readlines()
    C = coordinates(lines)
    clength = len(C)
    E = energy(lines)
    Comp = components(lines)
#    sComp = Isolvent(lines)
    d = dipmom(lines)
    hc = hirshcharge(lines)
    fv = freevol(lines)
    hv = hirshvol(lines)
    hdv = hirshdvec(lines)
    hdm = hirshdm(lines)
    hsm = hirshsm(lines)
    kse = kseig(lines)
    kse2 = []
    ilist = []
    for i in range(len(kse)):
        kse2.append(float(kse[i][-1]))
        if float(kse[i][0]) == 0.0:
           ilist.append(i)
#    c6p = c6pol(lines)
    ad = absdip(lines)
    qu = quadr(lines)
    gap = hlgap(lines)
    elqu = qu[0]
    ionqu = qu[1]
    totqu = qu[2]
    Fcomps = forcecomp(lines) 
    Fhf = Fcomps[0]
    Fion = Fcomps[1]
    Fm = Fcomps[2]
    Fhfe = Fcomps[3]
    Fpg = Fcomps[4]
    Fvdw = Fcomps[5]
    F = forces(lines)
    lock = 0
    if not C:
       print('Missing Coordinates!')
    else:
       lock += 1
    if E != 0:
       lock += 1
    else:
       print('Missing Energy!')
    if len(Comp) == 12:
       lock += 1
    else:
       print('Missing Energy Component(s)!')
    if len(d) == 3:
       lock += 1
    else:
       print('Missing Dipole Moment')
    if ad != 0:
       lock += 1
    else:
       print('Missing Absolute Dipole Moment')
    if len(hc) == clength:
       lock += 1
    else:
       print('Missing Hirshfeld Charge')
    if len(fv) == clength:
       lock += 1
    else:
       print('Missing Free Atom Volume')
    if len(hv) == clength:
       lock += 1
    else:
       print('Missing Hirshfeld Volume')
    if len(hdv) == clength:
       lock += 1
    else:
       print('Missing Hirshfeld Dipole Vector')
    if len(hdm) == clength:
       lock += 1
    else:
       print('Missing Hirshfeld Dipole Moment')
    if np.shape(hsm) == (clength, 3, 3):
       lock += 1
    else:
       print('Missing Hirshfeld Second Moment')
    if gap != 0.0:
       lock += 1
    else:
       print('Missing eigenvalues!')
#    if np.shape(c6p) == (clength, 3):
#       lock += 1
#    else:
#       print('Missing C6/Polarizability')
    if np.shape(qu) == (3,3):
       lock += 1
    else:
       print('Quardupole Missing')

    if np.shape(Fhf) == (clength, 3):
       lock += 1
    else:
       print('Missing Force Component')
    if np.shape(Fion) == (clength, 3):
       lock += 1
    else:
       print('Missing Force Component')
    if np.shape(Fhfe) == (clength, 3):
       lock += 1
    else:
       print('Missing Force Component')
    if np.shape(Fpg) == (clength, 3):
       lock += 1
    else:
       print('Missing Force Component')
    if np.shape(Fvdw) == (clength, 3):
       lock += 1
    else:
       print('Missing Force Component')
    if np.shape(F) == (clength, 3):
       lock += 1
    else:
       print('leo6')
       print('Missing Force Component')
    
    if lock == 19:
#vama        print ("Coordinates:")
#vama        print (C)
#vama        print ("Total energy (eV):")
#vama        print (E)
#vama        print ("Total energy components (eV):")
#vama        print ("Sum of eigenvaluess", "XC energy correction", "XC potential correction", "Free atom Elst", "Hartree correction", "MBD", "Kinetic Energy", "Electrostatic Energy", "Hartree-Fock Energy", "X energy", "C energy" , "Total XC energy")
#vama        print (Comp)
#vama        print ("Hirshfeld Charges:")
#vama        print (hc)
#vama        print ("Free Atom Volumes:")
#vama        print (fv)
#vama        print ("Hirshfeld Volumes:")
#vama        print (hv)
#vama        print ("Hirshfeld Dipole Vectors:")
#vama        print (hdv)
#vama        print ("Hirshfeld Dipole Moments:")
#vama        print (hdm)
#vama        print ("Hirshfeld Second Moments:")
#vama        print (hsm)
#vama        print ("Kohn-Sham Eigenvalues:")
#vama        print (kse2)
#vama        print ("HOMO-LUMO gap")
#vama        print (gap)
#vama        print ("C6 coefficients/ Atom Polarizabilities")
#vama        print (c6p)
#vama        print ("Total Dipole (eAng):")
#vama        print (d)
#vama        print ("Absolute Total Dipole (eAng):")
#vama        print (ad)
#vama        print ("Electronic Quadrupole Moment (eAng^2)")
#vama        print (elqu)
#vama        print ("Ionic Quadrupole Moment (eAng^2)")
#vama        print (ionqu)
#vama        print ("Total Quadrupole (eAng^2):")
#vama        print (totqu)
#vama        print ("Hellmann-Feynman Forces (eV/Ang):")
#vama        print (Fhf)
#vama        print ("Ionic Forces (eV/Ang):")
#vama        print (Fion)
#vama        print ("Multipole Forces (eV/Ang):")
#vama        print (Fm)
#vama        print ("Hartree-Fock Exchange Forces (eV/Ang):")
#vama        print (Fhfe)
#vama        print ("Pulay + GGA Forces (eV/Ang):")
#vama        print (Fpg)
#vama        print ("van der Waals Forces (eV/Ang):")
#vama        print (Fvdw)
#vama        print ("Total Forces (eV/Ang):")
#vama        print (F)   
        output={}
        output["Coordinates"]=C
        output["Total energy (eV):"]=E
        #output["Total energy components (eV):")
        output["SUM"]={}
        output["SUM"]["Sum of eigenvaluess"]=Comp[0]
        output["SUM"]["XC energy correction"]=Comp[1] 
        output["SUM"]["XC potential correction"]=Comp[2] 
        output["SUM"]["Free atom Elst"]=Comp[3] 
        output["SUM"]["Hartree correction"]=Comp[4] 
        output["SUM"]["MBD"]=Comp[5] 
        output["SUM"]["Kinetic Energy"]=Comp[6] 
        output["SUM"]["Electrostatic Energy"]=Comp[7] 
        output["SUM"]["Hartree-Fock Energy"]=Comp[8] 
        output["SUM"]["X energy"]=Comp[9] 
        output["SUM"]["C energy" ]=Comp[10] 
        output["SUM"]["Total XC energy"]=Comp[11]

#        output["Solvent"]={}
#        output["Solvent"]["Free Energy in Electrolyte"]=sComp[0]
#        output["Solvent"]["Surface Area of Cavity"]=sComp[1]
#        output["Solvent"]["Volume of Cavity"]=sComp[2]
#        output["Solvent"]["Nonelectrostatic Free Energy"]=sComp[3]
#        output["Solvent"]["Nonelstatic LPBE Solv Energy"]=sComp[4]

        output["Hirshfeld Charges:"]=hc
        output["Free Atom Volumes:"]=fv
        output["Hirshfeld Volumes:"]=hv
        output["Hirshfeld Dipole Vectors:"]=hdv
        output["Hirshfeld Dipole Moments:"]=hdm
        output["Hirshfeld Second Moments:"]=hsm
        output["Kohn-Sham Eigenvalues:"]=kse2
        output["HOMO energy"]=gap[-3]
        output["LUMO energy"]=gap[-2]
        output["HOMO-LUMO gap"]=gap[-1]
#        output["C6 coefficients/ Atom Polarizabilities"]=c6p
        output["Total Dipole (eAng)"]=d
        output["Absolute Total Dipole (eAng):"]=ad
        output["Electronic Quadrupole Moment (eAng^2)"]=elqu
        output["Ionic Quadrupole Moment (eAng^2)"]=ionqu
        output["Total Quadrupole (eAng^2):"]=totqu
        output["Hellmann-Feynman Forces (eV/Ang):"]=Fhf
        output["Ionic Forces (eV/Ang):"]=Fion
        output["Multipole Forces (eV/Ang):"]=Fm
        output["Hartree-Fock Exchange Forces (eV/Ang):"]=Fhfe
        output["Pulay + GGA Forces (eV/Ang):"]=Fpg
        output["van der Waals Forces (eV/Ang):"]=Fvdw
        output["Total Forces (eV/Ang):"]=F   
        ##print (output)
        with open('file.json','w') as fd:
              json.dump(output, fd, indent=4)

 
def coordinates(lines):
    active = False
    coordinates = []
    for line in lines:
        if not active:
          if "line of geometry.in" in line:
             active = True
             nl=0
#          print(line)
        else:
            nl+=1
            cc = re.findall("\-?\d+(?:\.\d+)\d+", line)
            cc2 = re.findall("\w+$", line)
            if len(cc) == 3:
              coordinates.append([cc2[0], float(cc[0]), float(cc[1]), float(cc[2]) ] )
            else:
                if ( nl > 7):
                   active = False

    return coordinates


def energy(lines):
    energy = 0.0
    for line in lines:
       if "| Total energy of the DFT / Hartree-Fock s.c.f. calculation" in line:
          energy=line.split()[-2]
    return energy

def hlgap(lines):
    hlgap = []
    for line in lines:
       if "Highest occupied state (VBM) at" in line:
          e1=line.split()[-2]
          hlgap.append(e1)
       if "Lowest unoccupied state (CBM) at" in line:
          e2=line.split()[-2]
          hlgap.append(e2)
       if "Overall HOMO-LUMO gap:" in line:
          e3=line.split()[-2]
          hlgap.append(e3)
    return hlgap


def components(lines):
    active = False
    components=[]
    for line in lines:
        if not active:
              if "Libmbd: Calculating MBD energy" in line:
                 active = True
        else:
            if "Sum of eigenvalues    " in line:
                soe = float(line.split()[-2])
                components.append(soe)
            if "XC energy correction" in line:
                xcec = float(line.split()[-2])
                components.append(xcec)
            if "XC potential correction" in line:
                xcpc = float(line.split()[-2])
                components.append(xcpc)
            if "Free-atom electrostatic energy" in line:
                faee = float(line.split()[-2])
                components.append(faee)
            if "Hartree energy correction" in line:
                hec = float(line.split()[-2])
                components.append(hec)
            if "vdW energy correction" in line:
                mbd = float(line.split()[-2])
                components.append(mbd)
            if "Kinetic energy" in line:
                ke = float(line.split()[-2])
                components.append(ke)
            if "Electrostatic energy" in line:
                elst = float(line.split()[-2])
                components.append(elst)
            if "Hartree-Fock Energy" in line:
                hfe = float(line.split()[-2])
                components.append(hfe)
            if "X Energy    " in line:
                xe = float(line.split()[-2])
                components.append(xe)
            if " C Energy    " in line:
                ce = float(line.split()[-2])
                components.append(ce)
            if "Total XC Energy" in line:
                txce = float(line.split()[-2])
                components.append(txce)
    return components

def Isolvent(lines):
    active = False
    scomponents=[]
    for line in lines:
        if not active:
              if "Libmbd: Calculating MBD energy" in line:
                 active = True
        else:
            if "Free Energy in Electrolyte" in line:
                p1 = float(line.split()[-2])
                scomponents.append(p1)
            if "Surface Area of Cavity" in line:
                p2 = float(line.split()[-2])
                scomponents.append(p2)
            if "Volume of Cavity" in line:
                p3 = float(line.split()[-2])
                scomponents.append(p3)
            if "Nonelectrostatic Free Energy" in line:
                p4 = float(line.split()[-2])
                scomponents.append(p4)
            if "Additional Nonelstatic LPBE Solvation Energy" in line:
                p5 = float(line.split()[-2])
                scomponents.append(p5)
    return scomponents

def forcecomp(lines):
    active = False
    hff = []
    ionf = []
    mf = []
    hfef = []
    pgf = []
    vdwf = []
    forcecomps=[]
    for line in lines:
        if not active:
              if "Libmbd: Calculating MBD energy" in line:
                 active = True
        else:
            if "Hellmann-Feynman " in line:
                hff.append([ float(line.split()[-3]), float(line.split()[-2]), float(line.split()[-1]) ])
            if "Ionic forces " in line:
                ionf.append([ float(line.split()[-3]), float(line.split()[-2]), float(line.split()[-1]) ])
            if "Multipole " in line:
                mf.append([ float(line.split()[-3]), float(line.split()[-2]), float(line.split()[-1]) ])
            if "Hartree-Fock exchange " in line:
                hfef.append([ float(line.split()[-3]), float(line.split()[-2]), float(line.split()[-1]) ])
            if "Pulay + GGA " in line:
                pgf.append([ float(line.split()[-3]), float(line.split()[-2]), float(line.split()[-1]) ])
            if "Van der Waals " in line:
                vdwf.append([ float(line.split()[-3]), float(line.split()[-2]), float(line.split()[-1]) ])
    forcecomps.append(hff)
    forcecomps.append(ionf)
    forcecomps.append(mf)
    forcecomps.append(hfef)
    forcecomps.append(pgf)
    forcecomps.append(vdwf)
    return forcecomps

def forces(lines):
    active = False
    forces =[]
    for line in lines:
       if not active:
          if " Total atomic forces (unitary forces cleaned) [eV/Ang]:" in line:
#             print(line)
             active = True
       else:
           ff = re.findall("\-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)", line)
           if len(ff) == 3:
              forces.append([float(ff[0]), float(ff[1]), float(ff[2])])
           else:
               active = False
#    print(forces)
    return forces

def dipmom(lines):
    dipole =[]
    for line in reversed(lines):
       if "| Total dipole moment [eAng]"  in line:
           dp = re.findall("\-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)", line)
           dipole=[float(dp[0]), float(dp[1]), float(dp[2])]
    return dipole

def absdip(lines):
    absdipole = 0.0
    for line in reversed(lines):
       if "| Absolute dipole moment"  in line:
           absdipole = float(line.split()[5])
    return absdipole

def quadr(lines):
    active = False
    quad =[]
    for line in lines:
       if not active:
          if "Quadrupole information:" in line:
              active = True
       else:
          if "electronic quadrupole moment" in line:
              eq = line.split()
          if "ionic quadrupole moment" in line:
              iq = line.split()
          if "Total quadrupole moment" in line:
              tq = line.split()
    quad = [ [float(eq[-3]), float(eq[-2]), float(eq[-1])], [float(iq[-3]), float(iq[-2]), float(iq[-1])], [float(tq[-3]), float(tq[-2]), float(tq[-1])] ]
    return quad



def hirshcharge(lines):
    active = False
    hcharge = []
    for line in lines:
       if not active:
          if "Performing Hirshfeld analysis of fragment charges and moments" in line:
              active = True
       else:
          if "Hirshfeld charge" in line:
              hh = float(re.findall("\-?\d+(?:\.\d+)", line)[0])
              hcharge.append(hh)
    return hcharge

def freevol(lines):
    active = False
    freevol = []
    for line in lines:
       if not active:
          if "Performing Hirshfeld analysis of fragment charges and moments" in line:
                active = True
       else:
          if "Free atom volume" in line:
             fav = float(re.findall("\-?\d+(?:\.\d+)", line)[0])
             freevol.append(fav)
    return freevol

def hirshvol(lines):
    active = False
    hirshvol = []
    for line in lines:
       if not active:
          if "Performing Hirshfeld analysis of fragment charges and moments" in line:
                active = True
       else:
          if "Hirshfeld volume" in line:
             hv = float(re.findall("\-?\d+(?:\.\d+)", line)[0])
             hirshvol.append(hv)
    return hirshvol


def hirshdvec(lines):
    active = False
    hirshdvec = []
    for line in lines:
       if not active:
          if "Performing Hirshfeld analysis of fragment charges and moments" in line:
                active = True
       else:
          if "Hirshfeld dipole vector" in line:
             lsp = line.split()
             hdv = [ float(lsp[-3]), float(lsp[-2]), float(lsp[-1]) ]
             hirshdvec.append(hdv)
    return hirshdvec

def hirshdm(lines):
    active = False
    hirshdm = []
    for line in lines:
       if not active:
          if "Performing Hirshfeld analysis of fragment charges and moments" in line:
                active = True
       else:
         if "Hirshfeld dipole moment" in line:
            lsp = line.split()
            hdm = float(re.findall("\-?\d+(?:\.\d+)", line)[0])
            #hdm = float(lsp[-1])
            hirshdm.append(hdm)
    return hirshdm


def hirshsm(lines):
    active1 = False
    active2 = False
    hirshsm = []
    for line in lines:
       if not active1:
          if "Performing Hirshfeld analysis of fragment charges and moments" in line:
                active1 = True
       else:
         if not active2:
            if "Hirshfeld second moments" in line:
               active2 = True
               lsp = line.split()
               hsms = []
               hsms.append([ float(lsp[-3]), float(lsp[-2]), float(lsp[-1]) ])
               
         else:
           lsp = line.split()
           if len(lsp) != 1:
               hsms.append([ float(lsp[-3]), float(lsp[-2]), float(lsp[-1]) ])
           else:
             hirshsm.append(hsms)
             active2 = False
    return hirshsm


def kseig(lines):
    active1 = False
    active2 = False
    kseigs = []
    nl=0
    for line in lines:
        if active1 == False:
             if "Change of forces" in line:
                 active1 = True
        else:
            if not active2:
                 if "Writing Kohn-Sham eigenvalues." in line:
                     active2 = True
            else:
                nl += 1
                kse = re.findall("\-?\d+(?:\.\d+)", line)
                if len(kse) == 3:
                    kseigs.append(kse)
                else:
                    if ( nl > 5):
                       active1 = False
    return kseigs
    
#def c6pol(lines):
#    active = False
#    c6pol =[]
#    c6p = []
#    for line in lines:
#       if not active:
#          if "Partitioned atomic C6 coefficients and polarizabilities" in line:
#             active = True
#             nl = 0
#       else:
#           nl += 1
#           c6pr = line.split()
#           if len(c6pr) == 6:
#              c6p.append( [c6pr[3],float(c6pr[4]),float(c6pr[5])] )
#           else:
#               if nl > 3:
#                  active = False
#    return c6p

def main(logfile):
      #print_geom_in (xyzfile)
      get_data (logfile)

if __name__ == "__main__":
      logfile=''
      logfile = argv[1]
      main(logfile)

