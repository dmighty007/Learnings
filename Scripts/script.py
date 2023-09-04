from pymol.cgo import *
import colorsys,sys,re
from pymol import cmd
import numpy as np
def mfunction(out,cage1,cage2,cage3):
    whole = out
    file1 = cage1
    file2 = cage2
    cmd.load(file1)
    cmd.load(file2)
    pref_file1 = cage1[:-4]
    pref_file2 = cage2[:-4]
    
    cmd.select("sol1",f"resn SOL and name OW and {pref_file1}")
    cmd.select("sol2",f"resn SOL and name OW and {pref_file2}")
    cmd.select("Me1",f"resn CBX and {pref_file1}")
    cmd.select("Me2",f"resn CBX and {pref_file2}")
    cmd.remove("Me1")
    cmd.remove("Me2")
    sol1_coords = cmd.get_coords("sol1",1)
    #print(sol1_coords.shape)
    from scipy.spatial import cKDTree
    #alist = []

    tree = cKDTree(sol1_coords)
    for i in range(len(sol1_coords)):
        cmd.select(f"sol1atom{i+1}",f"resname SOL and resid {i+1} and name OW and {pref_file1}")
    for i in range(len(sol1_coords)):
        indx  = tree.query(sol1_coords[i], k=4, distance_upper_bound = 3.5)[1][1:]
        indx = indx[~np.isin(indx, len(sol1_coords))]
        for j in indx:
            #if [i+1, j+1] not in alist and [j+1, i+1] not in alist:
            #    alist.append([i+1, j+1])
            cmd.bond(f"sol1atom{i+1}",f"sol1atom{j+1}")
        #cmd.delete(f"sol1atom{i+1}")
    sol2_coords = cmd.get_coords("sol2")
    tree = cKDTree(sol2_coords)
    for i in range(len(sol2_coords)):
        cmd.select(f"sol2atom{i+1}",f"resname SOL and resid {i+1} and name OW and {pref_file2}")
    for i in range(len(sol2_coords)):
        indx  = tree.query(sol2_coords[i], k=4, distance_upper_bound = 3.5)[1][1:]
        indx = indx[~np.isin(indx, len(sol2_coords))]
        for j in indx:
            #if [i+1, j+1] not in alist and [j+1, i+1] not in alist:
            #alist.append([i+1, j+1])
            cmd.bond(f"sol2atom{i+1}",f"sol2atom{j+1}")
        #cmd.delete(f"sol2atom{i+1}")
    for i in range(len(sol2_coords)):
        cmd.delete(f"sol2atom{i+1}")
    for i in range(len(sol1_coords)):
        cmd.delete(f"sol1atom{i+1}")
    try:
        file3 = cage3
        pref_file3 = cage3[:-4]
        cmd.load(cage3)
        cmd.select("sol3",f"resn SOL and name OW and {pref_file3}")
        cmd.select("Me3", f"resname CBX and {pref_file3}")
        cmd.remove("Me3")
        #md.select("edo3",f"resn DMSO and {pref_file3}")
        #cmd.hide("everything","edo3")

        sol3_coords = cmd.get_coords("sol3",1)
        tree = cKDTree(sol3_coords)
        for i in range(len(sol3_coords)):
            cmd.select(f"sol3atom{i+1}",f"resname SOL and resid {i+1} and name OW and {pref_file3}")
        for i in range(len(sol3_coords)):
            indx  = tree.query(sol3_coords[i], k=4, distance_upper_bound = 3.5)[1][1:]
            indx = indx[~np.isin(indx, len(sol3_coords))]
            for j in indx:
                #if [i+1, j+1] not in alist and [j+1, i+1] not in alist:
                    #alist.append([i+1, j+1])
                cmd.bond(f"sol3atom{i+1}",f"sol3atom{j+1}")
        cmd.show("sticks","sol3")
        cmd.color("red","sol3")
        for i in range(len(sol3_coords)):
            cmd.delete(f"sol3atom{i+1}")
        cmd.select("IPA3", f"resname IPA and  {pref_file3}")
        cmd.remove("IPA3")
        
    
    except :
        pass
    cmd.show("sticks","all")

    #cmd.hide("everything","allout")
    cmd.color("yellow", "sol1")
    cmd.color("blue", "sol2")
    cmd.set("orthoscopic","on")
    cmd.select("IPA1", f"resname IPA and  {pref_file1}")
    cmd.remove("IPA1")
    cmd.select("IPA2", f"resname IPA and  {pref_file2}")
    cmd.remove("IPA2")
    cmd.load(whole)
    pref_whole = whole[:-4]

    cmd.select("IPA", f"resname IPA and {pref_whole}")
    cmd.show("sticks", "IPA")
    cmd.select("ICE", f"resname ICE and {pref_whole}")
    #cmd.remove("ICE")
    cmd.select("Unnecessary", "resname MTS or resname MTL")
    cmd.show("spheres","Unnecessary")
    cmd.set("sphere_scale", 0.4)
    cmd.color("gray", "Unnecessary")
    cmd.select("water", f"resname SOL and {pref_whole}")
    #cmd.remove("water")
    cmd.set("stick_radius", 0.4, "IPA")
    cmd.select("CH4", f"resname CH4 or resname MTS or resname MTL and {pref_whole}")
    cmd.show("spheres", "CH4")
    cmd.color("gray", "CH4")
    cmd.set("sphere_scale", 0.4)
    cmd.rotate("y", 90,"all")
    return 0

    
cmd.extend("mfunction",mfunction)
