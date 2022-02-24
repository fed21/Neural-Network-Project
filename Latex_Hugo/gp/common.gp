#set term pdfcairo dashed enhanced font "Times,16" linewidth 2
set terminal postscript portrait enhanced color dashed lw 1 "Times" 16

set style line 1  dashtype 1 pt 1 ps 0.7 lc 1 lw 2
set style line 2  dashtype 1 pt 2 ps 0.7 lc 3 lw 2
set style line 3  dashtype 1 pt 3 ps 0.7 lc 4 lw 2

set style line 4  dashtype 1 pt 6 ps 0.7 lc 9 lw 2
set style line 5  dashtype 1 pt 4 ps 0.7 lc 7  lw 2
set style line 6  dashtype 1 pt 7 ps 0.7 lc 8  lw 2

set style line 7  dashtype 1 pt 8 ps 0.7 lc 10 lw 2
# set style line 2  dashtype 2 pt 2 lc 3 lw 1
# set style line 3  dashtype 4 pt 3 lc 4 lw 1

# set style line 4  dashtype 5 pt 6 lc 9 lw 1
# set style line 5  dashtype 3 pt 4 lc 7  lw 1


epsfile=sprintf("../figs/%s.eps", basename)
pdffile=sprintf("../figs/%s.pdf", basename)
tmpfile=sprintf("../figs/%s_tmp.pdf", basename)
set output epsfile

print "[@] ", pdffile, " [@]"
