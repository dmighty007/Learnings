function texcomplie()    
{    
  pdflatex $1.tex && bibtex $1.aux && pdflatex $1.tex && pdflatex $1.tex    
}

function pypaper ()
{
  python -m PyPaperBot --doi=$1 --dwn-dir=$2
}

function autoremove(){
	sudo pacman -Rs $(pacman -Qtdq)
}

#  Connecting to solaris ...

function solarbook(){
	ssh -L$1:localhost:$1 solaris -t ssh -L$1:localhost:$1 $2
}

function gluonbook(){
	ssh -L$1:localhost:$1 gluon -t ssh -L$1:localhost:$1 $2
}

#   Minicalculator on terminal ...
function qqbc(){
echo "scale=${2:-2}; $1" | bc -l
}

#   Random thing on terminal ...
function matrix(){
	perl -e '$|++; while (1) { print " ." x (rand(10) + 1), int(rand(2)) }'
}

#   To unzip or extract files ...
ex () {
    if [ -f $1 ] ; then
        case $1 in
            *.tar.bz2)   tar xjf $1        ;;
            *.tar.gz)    tar xzf $1     ;;
            *.bz2)       bunzip2 $1       ;;
            *.rar)       rar x $1     ;;
            *.gz)        gunzip $1     ;;
            *.tar)       tar xf $1        ;;
            *.tbz2)      tar xjf $1      ;;
            *.tgz)       tar xzf $1       ;;
            *.zip)       unzip $1     ;;
            *.Z)         uncompress $1  ;;
            *.7z)        7z x $1    ;;
            *)           echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# mkmv - creates a new directory and moves the file into it, in 1 step
# Usage: mkmv <file> <directory>
mkmv() {
    mkdir "$2"
    mv "$1" "$2"
}

# sanitize - set file/directory owner and permissions to normal values (644/755)
# Usage: sanitize <file>
sanitize() {
    chmod -R u=rwX,go=rX "$@"
    chown -R ${USER} "$@"
}

function_depends()  {
    search=$(echo "$1")
    sudo pacman -Sii $search | grep "Required" | sed -e "s/Required By     : //g" | sed -e "s/  /\n/g"
    }

wallpaper ()
{
 swaybg -m fill -i $1 & disown  
}

