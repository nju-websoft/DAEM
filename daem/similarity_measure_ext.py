import py_stringmatching as sm
from py_stringmatching import utils
from py_stringmatching.similarity_measure.token_similarity_measure import TokenSimilarityMeasure


class SimilarityCombination():
    def __init__(self, tk, sm):
        self.tokenizer = tk
        self.metric = sm
    def __call__(self, x, y):
        return self.metric.get_raw_score(self.tokenizer.tokenize(x), self.tokenizer.tokenize(y))


class EuclideanDistance(TokenSimilarityMeasure):
    def __init__(self):
        super(EuclideanDistance, self).__init__()

    def get_raw_score(self, set1, set2):
        from collections import Counter
        from math import sqrt
        utils.sim_check_for_none(set1, set2)
        utils.sim_check_for_list_or_set_inputs(set1, set2)
        if utils.sim_check_for_exact_match(set1, set2):
            return 1.0
        if utils.sim_check_for_empty(set1, set2):
            return 0
        
        vec1, vec2 = Counter(set1), Counter(set2)
        
        dist, l1, l2 = 0.0, 0.0, 0.0
        for tok in vec1.keys() | vec2.keys():
            f1 = vec1.get(tok, 0)
            f2 = vec2.get(tok, 0)
            dist += (f1 - f2) * (f1 - f2)
            l1 += f1 * f1
            l2 += f2 * f2
        return sqrt(dist / l1 / l2)

    def get_sim_score(self, set1, set2):
        return self.get_raw_score(set1, set2)


class BlockDistance(TokenSimilarityMeasure):
    def __init__(self):
        super(BlockDistance, self).__init__()

    def get_raw_score(self, set1, set2):
        from collections import Counter
        from math import sqrt
        utils.sim_check_for_none(set1, set2)
        utils.sim_check_for_list_or_set_inputs(set1, set2)
        if utils.sim_check_for_exact_match(set1, set2):
            return 1.0
        if utils.sim_check_for_empty(set1, set2):
            return 0
        
        vec1, vec2 = Counter(set1), Counter(set2)
        
        dist, l1, l2 = 0.0, 0.0, 0.0
        for tok in vec1.keys() | vec2.keys():
            f1 = vec1.get(tok, 0)
            f2 = vec2.get(tok, 0)
            dist += abs(f1 - f2)
            l1 += f1 * f1
            l2 += f2 * f2
        return dist / sqrt(l1 * l2)

    def get_sim_score(self, set1, set2):
        return self.get_raw_score(set1, set2)


metrics = {
    'cosineSimilarity': SimilarityCombination(sm.WhitespaceTokenizer(return_set=False), sm.Cosine()),
    'jaccard': SimilarityCombination(sm.WhitespaceTokenizer(return_set=True), sm.Jaccard()),
    'generalizedJaccard': SimilarityCombination(sm.WhitespaceTokenizer(return_set=False), sm.Jaccard()),
    'dice': SimilarityCombination(sm.WhitespaceTokenizer(return_set=True), sm.Dice()),
    'simonWhite': SimilarityCombination(sm.WhitespaceTokenizer(return_set=True), sm.Dice()),
    'overlapCoefficient': SimilarityCombination(sm.WhitespaceTokenizer(return_set=True), sm.OverlapCoefficient()),
    'levenshtein': sm.Levenshtein().get_raw_score,
    'jaro': sm.Jaro().get_sim_score,
    'jaroWinkler': sm.JaroWinkler().get_raw_score,
    'mongeElkan': SimilarityCombination(sm.WhitespaceTokenizer(return_set=False), sm.MongeElkan(sim_func=sm.SmithWaterman().get_raw_score)),
    'needlemanWunch': sm.NeedlemanWunsch().get_raw_score,
    'smithWaterman': sm.SmithWaterman().get_raw_score,
    'soundex': sm.Soundex().get_raw_score,
    'identity': lambda x, y: 1.0 if x == y else 0.0,
    'euclideanDistance': SimilarityCombination(sm.WhitespaceTokenizer(return_set=False), EuclideanDistance()),
    'blockDistance': SimilarityCombination(sm.WhitespaceTokenizer(return_set=False), BlockDistance()),
    'qGramsDistance': SimilarityCombination(sm.QgramTokenizer(qval=3), BlockDistance()),
}

if False:
    for k, v in metrics.items():
        print(k, v('Robert', 'Rupert'))

# TODO: dice and etc, for set or multiset
# damerauLevenshtein
# smithWatermanGotoh
# longestCommonSubsequence
# longestCommonSubstring