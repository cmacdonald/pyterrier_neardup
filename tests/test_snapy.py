import unittest
import pyterrier as pt
import pandas as pd
import tempfile
import shutil
class TestSnapy(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not pt.started():
            pt.init()

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)


    def test_vaswani_mem(self):
        from pyterrier_neardup.snapy_lsh import SnapyMinHash
        self._test_vaswani(SnapyMinHash(), TWO_D=False)

    def _test_vaswani(self, mhs, TWO_D = False):
        corpus_file = pt.datasets.get_dataset("vaswani").get_corpus()[0]
        Arrays = pt.autoclass("java.util.Arrays")
        corpus_file_list = Arrays.asList(corpus_file)
        trec_properties = {
            "TrecDocTags.doctag": "DOC",
            "TrecDocTags.idtag": "DOCNO",
            "TrecDocTags.skip": "DOCHDR",
            "TrecDocTags.casesensitive": "false",
            "trec.collection.class": "TRECCollection",
        }
        for k,v in trec_properties.items():
            pt.ApplicationSetup.setProperty(k, v)
        corpus = pt.autoclass("org.terrier.indexing.TRECCollection")(
            corpus_file_list, 
            pt.autoclass("org.terrier.utility.TagSet").TREC_DOC_TAGS, "", "")

        def _get_text(d):
            terms =[]
            while not d.endOfDocument():
                t = d.getNextTerm()
                if t is None:
                    continue
                terms.append(t)
            return " ".join(terms)
        
        def _corpus_iter():
            while corpus.nextDocument():
                doc = corpus.getDocument()
                text = _get_text(doc)
                docno = doc.getProperty("docno")
                yield docno, text

        mhs.index(_corpus_iter())

        
        if TWO_D:
            oneDmatrix = mhs.pairwise_sim()
            maxPos=None
            maxSim=-1
            numDocs = len(mhs.docNames)
            for i in range(0, numDocs):     
                # For each of the other test documents...
                for j in range(i + 1, numDocs):
                    if oneDmatrix[mhs.getTriangleIndex(i, j)] > maxSim:
                        maxSim = oneDmatrix[mhs.getTriangleIndex(i, j)]
                        maxPos = (i,j)
            print("Most similar pair is %s, with sim %f" % (str(maxPos), maxSim))

        input = pd.DataFrame([["17"]], columns=["docno"])
        rtr = mhs.transform(input)
        print(rtr)