from snapy import MinHash, LSH
from tqdm import tqdm
import pandas as pd

class SnapyMinHash():

    def __init__(self):
        pass

    def index(self, iter):
        # Create MinHash object.
        dummy = " ".join(["dummy"]*9)
        minhash = MinHash([dummy], n_gram=9, permutations=100, hash_bits=64, seed=3)

        # Create LSH model.
        self.lsh = LSH(minhash, [""], no_of_bands=50)

        for docid, (docno, text) in enumerate(iter):
            new_minhash = MinHash([text], n_gram=9, permutations=100, hash_bits=64, seed=3)
            self.lsh.update(new_minhash, [docno])
            if docid % 1000 == 0:
                print(docid)

    def transform(self, input):
        docno_provided = "docno" in input.columns
        assert docno_provided
        rtr = []
        for row in tqdm(input.itertuples(), total=len(input)):
            docno = row.docno
            for r in self.lsh.query(docno):
                rtr.append( [docno, r, 1] )
        return pd.DataFrame(rtr, columns=["docno_x", "docno_y", "score"])

