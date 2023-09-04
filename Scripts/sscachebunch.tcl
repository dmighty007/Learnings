#VMD  --- start of VMD description block
#Name:
# SSCacheBunch
#Synopsis:
# Automatically stores secondary structure information for animations with tweaked performance
#Version:
# 1.0
#Uses VMD version:
# 1.1
#Ease of use:
# 2
#Procedures:
# <li>start_sscache molid - start caching the given molecule
# <li>stop_sscache molid - stop caching
# <li>reset_sscache - reset the cache
# <li>sscache - internal function used by trace
#Description:
# Calculates and stores the secondary structure assignment for 
# each timestep.  This lets you see how the secondary structure
# changes over a trajectory.
# <p>
# It is turned on with the command "start_sscache> followed by the
# molecule number of the molecule whose secondary structure should be
# saved (the default is "top", which gets converted to the correct
# molecule index).  Whenever the frame for that molecule changes, the
# procedure "sscache" is called.
# <p>
#   "sscache" is the heart of the script.  It checks if a secondary
# structure definition for the given molecule number and frame already
# exists in the Tcl array sscache_data(molecule,frame).  If so, it uses
# the data to redefine the "structure" keyword values (but only for
# the protein residues).  If not, it calls the secondary structure
# routine to evaluate the secondary structure based on the new
# coordinates.  The results are saved in the sscache_data array.
# <p>
# Once the secondary structure values are saved, the molecule can be
# animated rather quickly and the updates can be controlled by the
# animate form.
# <p>
#  To turn off the trace, use the command "stop_sscache", which
# also takes the molecule number.  There must be one "stop_sscache"
# for each "start_sscache".  The command "clear_sscache" resets
# the saved secondary structure data for all the molecules and all the
# frames.
# <p>
#Performance tweak SSCACHEBUNCH (linux only)
# Frames are computed in a forward bunch to save TCL "exec" calls which
# are the time consuming factor, not STRIDE.
# Use env(SSCACHEBUNCH) to control how many frames are computed at once.
# Use env(SSCACHEPARA) to control how many STRIDEs are called in parallel.
# Multiple thousands are possible for BUNCH value, only limit is hard disk space.
# 1st play of trajectory shoulnd't be reverse direction of course.
#Files: 
# <a href="sscachebunch.vmd">sscachebunch.vmd</a>
#See also:
# the VMD user's guide
#Author: 
# Andrew Dalke &lt;dalke@ks.uiuc.edu&gt;
# Norman Geist norman.geist@uni-greifswald.de
#Url: 
# http://www.ks.uiuc.edu/Research/vmd/script_library/sscachebunch/
#\VMD  --- end of block


# start the cache for a given molecule
proc start_sscache {{molid top}} {
    global sscache_data
    if {! [string compare $molid top]} {
	set molid [molinfo top]
    }
    global vmd_frame
    # set a trace to detect when an animation frame changes
    trace variable vmd_frame($molid) w sscache
    return
}

# remove the trace (need one stop for every start)
proc stop_sscache {{molid top}} {
    if {! [string compare $molid top]} {
	set molid [molinfo top]
    }
    global vmd_frame
    trace vdelete vmd_frame($molid) w sscache
    return
}


# reset the whole secondary structure data cache
proc reset_sscache {} {
    global sscache_data
    if [info exists sscache_data] {
        unset sscache_data
    }
    return
}

# when the frame changes, trace calls this function...
set env(SSCACHEBUNCH) 500; #how many frames to bunch out to save exc calls
set env(SSCACHEPARA) 16; #how many STRIDEs to call at once
set env(SSRAND) [expr int(rand()*1e6)]; #for safely running multiple VMDs
proc sscache {name index op} {
    # name == vmd_frame
    # index == molecule id of the newly changed frame
    # op == w
    
    global sscache_data
    global env

    # get the protein CA atoms
    set sel [atomselect $index "protein"]
    set resids [lsort -unique -integer -increasing [$sel get resid]]
    set num [llength $resids]

    ## get the new frame number
    # Tcl doesn't yet have it, but VMD does ...
    set frame [molinfo $index get frame]
    set frames [molinfo $index get numframes]

    # see if the ss data exists in the cache
    if [info exists sscache_data($index,$frame)] {
	$sel set structure $sscache_data($index,$frame)
	$sel delete
	return
    }

    # doesn't exist, so (re)calculate it
    #vmd_calculate_structure $index; # <-- the old call

    # bunch the execution
    set fp [open ${env(TMPDIR)}/sscachebunch${env(SSRAND)}.sh "w"]
    for {set i 0} {$i < $env(SSCACHEBUNCH)} {incr i} {
      if {[expr $frame+$i] >= $frames} {break;}
      $sel frame [expr $frame+$i]
      $sel writepdb ${env(TMPDIR)}/tmp${env(SSRAND)}_${index}_${i}.pdb
      puts $fp "${env(STRIDE_BIN)}  ${env(TMPDIR)}/tmp${env(SSRAND)}_${index}_${i}.pdb | grep -E '^STR' > ${env(TMPDIR)}/tmp${env(SSRAND)}_${index}_${i}.out&"
      if {[expr $i % $env(SSCACHEPARA) == 0]} {#do env(SSCACHEPARA) at once/in parallel
	puts $fp "wait \$(jobs -p)"
      }
    }
    puts $fp "wait \$(jobs -p)"
    close $fp
    catch {exec -ignorestderr /bin/bash < ${env(TMPDIR)}/sscachebunch${env(SSRAND)}.sh} err

    # read and parse stride output
    for {set i 0} {$i < $env(SSCACHEBUNCH)} {incr i} {
      if {[expr $frame+$i] >= $frames} {break;}
      $sel frame [expr $frame+$i]

      set fp [open ${env(TMPDIR)}/tmp${env(SSRAND)}_${index}_${i}.out "r"]
      set raw [read $fp]
      set lines [split $raw "\n"]
      set STR ""
      set remain $num
      foreach line $lines {
	set lstr [string map {" " "C"} [string range $line 10 [expr $remain > 50 ? 50+10-1 : $remain+10-1]]]
	set remain [expr $remain-[string length $lstr]]
	set STR "${STR}${lstr}"
      }
      close $fp

      file delete ${env(TMPDIR)}/tmp${env(SSRAND)}_${index}_${i}.out
      file delete ${env(TMPDIR)}/tmp${env(SSRAND)}_${index}_${i}.pdb

      # apply the structure
      if {[string length $raw] > 0} {
# this was slow
# 	set j 0
# 	foreach resid $resids {
# 	  set sel2 [atomselect $index "resid $resid and name CA"]
# 	  $sel2 frame [expr $frame+$i]
# 	  $sel2 set structure [string index $STR $j]
# 	  $sel2 delete
# 	  incr j
# 	}
# this is incredible fast, but only CA are threated <- safe?
	set sel2 [atomselect $index "resid $resids and name CA"]
	$sel2 frame [expr $frame+$i]
	$sel2 set structure [split $STR ""]
	$sel2 delete
      } else {
	$sel set structure "C"
      }

      # save the data for next time
      set sscache_data($index,[expr $frame+$i]) [$sel get structure]
    }

    $sel set structure $sscache_data($index,$frame)
    $sel delete
    return
}

# start sscache for all molecules loaded
# -> nice for remd
proc start_sscache_all {} {
  foreach mol [molinfo list] {
    start_sscache $mol
  }
}

# stop sscache for all molecules
proc stop_sscache_all {} {
  foreach mol [molinfo list] {
    stop_sscache $mol
  }
}
