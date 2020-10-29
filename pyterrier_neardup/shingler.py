import binascii
import time
import random
import pandas as pd
from tqdm import tqdm
import base64
from typing import List
import more_itertools

# Record the maximum shingle ID that we assigned.
maxShingleID = 2**32-1

# We need the next largest prime number above 'maxShingleID'.
# I looked this value up here: 
# http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
nextPrime = 4294967311

# from pyterrier_bert
def slidingWindow(sequence, winSize, step):
    return [x for x in list(more_itertools.windowed(sequence,n=winSize, step=step)) if x[-1] is not None]


def shingles(text):
  tokens = text.split(" ")
  four_grams = slidingWindow(tokens, 4, 1)
  shingles =[" ".join(gram) for gram in four_grams]
  hashed_shingles = set([ binascii.crc32(shingle.encode("utf-8")) & 0xffffffff for shingle in shingles])
  return hashed_shingles, len(tokens) -2

# Generate a list of 'k' random coefficients for the random hash functions,
# while ensuring that the same value does not appear multiple times in the 
# list.
def pickRandomCoeffs(k):
  # Create a list of 'k' random values.
  randList = []
  
  while k > 0:
    # Get a random shingle ID.
    randIndex = random.randint(0, maxShingleID) 
  
    # Ensure that each random number is unique.
    while randIndex in randList:
      randIndex = random.randint(0, maxShingleID) 
    
    # Add the random number to the list.
    randList.append(randIndex)
    k = k - 1
    
  return randList

class MinHashShingler:

    def __init__(self, num_hashes=10, verbose=0, sim_threshold=0.9):
        self.numHashes = 10
        self.verbose = verbose
        self.sim_threshold = sim_threshold

        # Our random hash function will take the form of:
        #   h(x) = (a*x + b) % c
        # Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
        # a prime number just greater than maxShingleID.

        # For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.   
        self.coeffA = pickRandomCoeffs(self.numHashes)
        self.coeffB = pickRandomCoeffs(self.numHashes)

    # Define a function to map a 2D matrix coordinate into a 1D index.
    def getTriangleIndex(self, i, j):
        numDocs = len(self.docNames)
        # If i == j that's an error.
        if i == j:
            raise ValueError("Can't access triangle matrix with i == j")
        # If j < i just swap the values.
        if j < i:
            temp = i
            i = j
            j = temp
        
        # Calculate the index within the triangular array.
        # This fancy indexing scheme is taken from pg. 211 of:
        # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
        # But I adapted it for a 0-based index.
        # Note: The division by two should not truncate, it
        #       needs to be a float. 
        k = int(i * (numDocs - (i + 1) / 2.0) + j - i) - 1
        
        return k

    def get_signature(self, text) -> List[int]:
        shinglesInDoc, tokens = shingles(text)
        self.totalShingles += tokens -2

        # The resulting minhash signature for this document. 
        signature = []
        if len(shinglesInDoc) == 0:
            self.emptyDocs += 1
            return signature
        
        # For each of the random hash functions...
        for i in range(0, self.numHashes):
            # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
            # the maximum possible value output by the hash.
            minHashCode = nextPrime + 1

            for shingleID in shinglesInDoc:
                # Evaluate the hash function.
                hashCode = (self.coeffA[i] * shingleID + self.coeffB[i]) % nextPrime 
                
                # Track the lowest hash code seen.
                if hashCode < minHashCode:
                    minHashCode = hashCode
            signature.append(minHashCode)
        return signature


    def index(self, iter):


        docNames = []
        self.emptyDocs = 0
        self.totalShingles = 0
        t0 = time.time()

        signatures = []
        for docno, text in iter:
            docNames.append(docno)
            signatures.append(self.get_signature(text))
            
        numDocs = len(docNames)
        print('\nShingling ' + str(numDocs) + ' docs took %.2f sec.' % (time.time() - t0))
        print('\nAverage shingles per doc: %.2f' % (self.totalShingles / numDocs))


        # Calculate the elapsed time (in seconds)
        elapsed = (time.time() - t0)
        self.signatures = signatures
        self.docNames = docNames
        self.numDocs = len(docNames)
        self.docName2id = { docno : i for i, docno in enumerate(docNames)}
        print("\nGenerating MinHash signatures took %.2fsec" % elapsed) 

    def transform(self, input):
        threshold = self.sim_threshold
        docid_provided = "docid" in input.columns
        docno_provided = "docno" in input.columns
        assert docid_provided or docno_provided

        numDocs = len(self.docNames)
        rtr = []
        with tqdm(total=numDocs * len(input)) as pbar:
            for i, row in enumerate(input.itertuples()):
                if docid_provided:
                    docid = row.docid
                    docno = self.docNames[docid]
                else:
                    docid = self.docName2id[row.docno]
                    docno = row.docno

                signature1 = self.signatures[docid]
                if len(signature1) == 0:
                    # empty doc
                    continue
                        
                # For each of the other test documents...
                for j in range(numDocs):
                    if docid == j:
                        continue
                    
                    # Get the MinHash signature for document j.
                    signature2 = self.signatures[j]
                    if len(signature2) == 0:
                        # empty doc
                        continue
                    
                    count = 0
                    # Count the number of positions in the minhash signature which are equal.
                    for k in range(0, self.numHashes):
                        count += (signature1[k] == signature2[k])
                    
                    # Record the percentage of positions which matched.    
                    estJsim = (count / self.numHashes)
                    if estJsim > threshold:
                        rtr.append([docno, docid, self.docNames[j], j, estJsim])
                    pbar.update(1)
        
        return pd.DataFrame(rtr, columns=["docno_x", "docid_x", "docno_y", "docid_x", "score"])

    def pairwise_sim(self):
        numDocs = self.numDocs
        print('\nComparing all signatures for %d docs...' % numDocs)        
        numElems = int(numDocs * (numDocs - 1) / 2)
        
        # Creates a N x N triangular matrix initialized to 0.
        estJSim = [0 for x in range(numElems)]
  
        # Time this step.
        t0 = time.time()

        with tqdm(total=numDocs * (numDocs / 2)) as pbar:
            # For each of the test documents...
            for i in range(0, numDocs):
                # Get the MinHash signature for document i.
                signature1 = self.signatures[i]
                    
                # For each of the other test documents...
                for j in range(i + 1, numDocs):
                    
                    # Get the MinHash signature for document j.
                    signature2 = self.signatures[j]
                    
                    count = 0
                    # Count the number of positions in the minhash signature which are equal.
                    for k in range(0, self.numHashes):
                        count += (signature1[k] == signature2[k])
                    
                    # Record the percentage of positions which matched.    
                    estJSim[self.getTriangleIndex(i, j)] = (count / self.numHashes)
                    pbar.update(1)

        # Calculate the elapsed time (in seconds)
        elapsed = (time.time() - t0)
        print("\nComparing MinHash signatures took %.2fsec" % elapsed)
        return estJSim


def shingles2text(sigs):
    for s in sigs:
        assert s < maxShingleID, "s is %d maxShingleID is %d" % (s, maxShingleID)
    return " " .join([ base64.b32encode(sig.to_bytes(4, 'big'))[0:6].decode('ascii') + str(i)  for i, sig in enumerate(sigs)])


from pyterrier.index import IterDictIndexer, IndexingType
class IndexingOfShingles(MinHashShingler):

    def __init__(self, index_loc, type=IndexingType.MEMORY, **kwargs):
        super().__init__(**kwargs)
        self.index_loc = index_loc
        self.index_type = type

    def pairwise_sim(self):
        assert False, "Not supported by IndexingOfShingles"

    def index(self, iter):
        import pyterrier as pt
        
        indexer = IterDictIndexer(self.index_loc, type=self.index_type)
        indexer.setProperties(**{'indexer.meta.reverse.keys':'docno'})
        self.totalShingles = 0
        self.emptyDocs = 0
        
        def generator():
            for docid, (docno, text) in enumerate(iter):
                signature = self.get_signature(text)
                if docid % 1000 == 0:
                    print(docid)
                yield { 'docno' : docno, 'text' : shingles2text(signature) }
        self.indexref = indexer.index(generator())
        index = pt.IndexFactory.of(self.indexref)
        print(index.getCollectionStatistics().toString())
        assert "docno" in index.getMetaIndex().getReverseKeys()
        index.close()

    def get_query(self, index, docid):
        die = index.getDocumentIndex().getDocumentEntry(docid)
        if die.getDocumentLength() == 0:
            return ""
        lex = index.getLexicon()
        ip = index.getDirectIndex().getPostings(die)
        terms = [lex.getLexiconEntry(p.getId()).getKey() for p in ip]
        # check we have expected number of hashes
        assert len(terms) == self.numHashes, "Mismatch on postings, expected %d, found %d for docid %d, doclen %d: %s" % (self.numHashes, len(terms), docid, die.getDocumentLength(), str(terms))
        return " ".join(terms)

    def transform(self, input):
        threshold = 0.9
        docid_provided = "docid" in input.columns
        docno_provided = "docno" in input.columns
        assert docid_provided or docno_provided
        import pyterrier as pt
        from pyterrier import autoclass
        index = pt.IndexFactory.of(self.indexref)

        for k,v in pt.BatchRetrieve.default_properties.items():
            pt.ApplicationSetup.setProperty(k,v)
        
        ManagerFactory = autoclass("org.terrier.querying.ManagerFactory")
        manager = ManagerFactory._from_(self.indexref)

        rtr = []
        for row in tqdm(input.itertuples(), total=len(input)):
            if docid_provided:
                docid = row.docid
                if not docno_provided:
                    docno = index.getMetaIndex().getItem("docno", docid)
            else:
                docno = row.docno
                docid = index.getMetaIndex().getDocument("docno", docno)
                if docid == -1:
                    raise KeyError("Could not convert docno %s to a docid" % docno)

            q = self.get_query(index, docid)
            srq = manager.newSearchRequest(docno, q)
            srq.setControl("wmodel", "Tf")
            manager.runSearchRequest(srq)
            results = srq.getResults()
            for r in results:
                score = r.getScore() / self.numHashes
                if score > self.sim_threshold and r.getDocid() != docid:
                    rtr.append([docno, docid, r.getMetadata("docno"), r.getDocid(), score])
        return pd.DataFrame(rtr, columns=["docno_x", "docid_x", "docno_y", "docid_x", "score"])

