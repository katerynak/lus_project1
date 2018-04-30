from subprocess import call, check_output
import os
from fileNames import FileNames

class ConceptTagger(object):
    directory = FileNames.FST_DIR.value

    def __init__(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    #returns name of acceptor file
    def create_acceptor(self, string, fileLex):
        os.system('echo %r > str' % string)
        call('cat str | farcompilestrings --symbols={0} --unknown_symbol=\'<unk>\' --generate_keys=1 '
             '--keep_symbols=true | farextract --filename_suffix=\'.fst\''.format(fileLex),
             shell=True)
        call('rm str', shell=True)
        return '1.fst'

    def create_unigram_tagger(self, lexFile, inputFile, outFile):
        call('fstcompile --isymbols={0} --osymbols={0} {1} > {2}'.format(lexFile, inputFile, outFile), shell=True)

    def create_language_model(self, lexFile, inputFile, outFile, smoothing = "witten_bell", order=3):
        call('farcompilestrings --symbols={0} --unknown_symbol=\'<unk>\' {1} > tmp.far'.format(lexFile, inputFile),
             shell=True)
        call('ngramcount --order={0} --require_symbols=false tmp.far > tmp.cnt'.format(order),
             shell=True)
        call('ngrammake --method={0} tmp.cnt > {1}'.format(smoothing, outFile), shell=True)
        call('rm tmp.cnt; rm tmp.far', shell=True)

    def composeFsts(self, fst1File, fst2File, fstOutFile):
        call('fstcompose {0} {1} > {2}'.format(fst1File, fst2File, fstOutFile), shell=True)

    def shortestPath(self, fstInFile, fstOutFile):
        call('fstrmepsilon {0} | fstshortestpath > {1}'.format(fstInFile, fstOutFile), shell=True)

    def parseOut(self, lexFile, fstInFile):
        out = check_output('fsttopsort {1} | fstprint -osymbols={0} | cut -f4 | grep .'.format(lexFile, fstInFile),
                                      shell=True).decode("utf-8")
        return out

