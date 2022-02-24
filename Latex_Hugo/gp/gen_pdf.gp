
set output '/dev/null'

cmdconvert = sprintf("epspdf %s %s",   \
                   epsfile, pdffile)
print cmdconvert
system (cmdconvert)

cmdcrop = sprintf("pdfcrop %s %s && mv %s %s",   \
                   pdffile, tmpfile, tmpfile, pdffile)
print cmdcrop

system (cmdcrop)
