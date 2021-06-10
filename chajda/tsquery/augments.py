'''
'''

from chajda.tsvector import lemmatize, Config
from chajda.tsquery.__init__ import to_tsquery

def augments_gensim(lang, word, config=Config(), n=5):
    '''
    Returns n words that are "similar" to the input word in the target language.
    These words can be used to augment a search with the Query class.

    >>> to_tsquery('en', 'baby boy', augment_with=augments_gensim)
    '(baby:A | boy:B | girl:B | newborn:B | pregnant:B | mom:B) & (boy:A | girl:B | woman:B | man:B | kid:B | mother:B)'

    >>> to_tsquery('en', '"baby boy"', augment_with=augments_gensim)
    'baby:A <1> boy:A'

    >>> to_tsquery('en', '"baby boy" (school | home) !weapon', augment_with=augments_gensim)
    '(baby:A <1> boy:A) & ((school:A | college:B | campus:B | graduate:B | elementary:B | student:B) | (home:A | leave:B | rest:B | come:B)) & !(weapon:A | explosive:B | weaponry:B | gun:B | ammunition:B | device:B)'

    >>> augments_gensim('en','baby', n=5)
    ['boy', 'girl', 'newborn', 'pregnant', 'mom']

    >>> augments_gensim('en','school', n=5)
    ['college', 'campus', 'graduate', 'elementary', 'student']

    >>> augments_gensim('en','weapon', n=5)
    ['explosive', 'weaponry', 'gun', 'ammunition', 'device']
    '''

    # load the model if it's not already loaded
    try:
        augments_gensim.model
    except AttributeError:
        import gensim.downloader
        with suppress_stdout_stderr():
            augments_gensim.model = gensim.downloader.load("glove-wiki-gigaword-50")

    # find the most similar words;
    try:
        topn = augments_gensim.model.most_similar(word, topn=n+1)
        words = ' '.join([ word for (word,rank) in topn ])

    # gensim raises a KeyError when the input word is not in the vocabulary;
    # we return an empty list to indicate that there are no similar words
    except KeyError:
        return []

    # lemmatize the results so that they'll be in the search document's vocabulary
    words = lemmatize(lang, words, add_positions=False, config=config).split()
    words = list(filter(lambda w: len(w)>1 and w != word, words))[:n]

    return words


import fasttext
import fasttext.util
from annoy import AnnoyIndex

fasttext_models = {}
annoy_indices = {}

def load_index(lang, model, index):
    '''
    This function populates an AnnoyIndex with vectors from a fasttext model, saves the AnnoyIndex to disk, and then loads from disk.
    '''
    i = 0
    for j in model[lang].words:
        v = model[lang][j]
        index[lang].add_item(i,v)
        i += 1
    index[lang].build(10)
    index[lang].save('{0}.ann'.format(lang))
    index[lang].load('{0}.ann'.format(lang))

def load_model(lang):
    '''
    This function downloads a fasttext model for a particular language, and returns the loaded model.
    '''
    with suppress_stdout_stderr():
        fasttext.util.download_model(lang, if_exists='ignore')
    return fasttext.load_model('cc.{0}.300.bin'.format(lang))

def augments_fasttext(lang, word, config=Config(), n=5, annoy=True):
    ''' 
    Returns n words that are "similar" to the input word in the target language.
    These words can be used to augment a search with the Query class.
    This function will default to using the annoy library to get the nearest neighbors.
    Set annoy=False to use fasttext library for nearest neighbor query.

    #>>> to_tsquery('en', 'baby boy', augment_with=augments_fasttext)
    #'(baby:A | newborn:B | mamamade:B | postbath:B | bride:B) & (boy:A | lad:B | man:B | boyman:B | boylike:B)'
    
    #'(baby:A | newborn:B | infant:B) & (boy:A | girl:B | boyhe:B | boyit:B)'

    #>>> to_tsquery('en', '"baby boy"', augment_with=augments_fasttext)
    #'baby:A <1> boy:A'

    #>>> to_tsquery('en', '"baby boy" (school | home) !weapon', augment_with=augments_fasttext)
    #'(baby:A <1> boy:A) & ((school:A | college:B | highschool:B | 7thgraders:B) | (home:A | office:B | hospital:B | return:B)) & !(weapon:A | pistol:B | arsenal:B | rifle:B | minigun:B)'

    #'(baby:A <1> boy:A) & ((school:A | schoo:B | schoolthe:B | schoool:B | kindergarten:B) | (home:A | house:B | homethe:B | homewhen:B | homethis:B)) & !(weapon:A | weaponthe:B | weopon:B)'

   # >>> augments_fasttext('en','weapon', n=5, annoy=False)
   # ['weaponthe', 'weopon']

   # >>> augments_fasttext('en','king', n=5, annoy=False)
   # ['queen', 'kingthe']

   # >>> augments_fasttext('en','weapon', n=5)
   # ['pistol', 'arsenal', 'rifle', 'minigun']

    >>> augments_fasttext('en','king', n=5)
    ['throne', 'prince', 'kingship', 'kingdom']

    NOTE:
    Due to the size of fasttext models (>1gb),
    testing multiple languages in the doctest requires more space than github actions allows for.
    For this reason, tests involving languages other than English have been commented out below.

    #>>> augments_fasttext('ja','さようなら', n=5)
    #['さよなら', 'バイバイ', 'サヨウナラ', 'さらば', 'おしまい']

    #>>> augments_fasttext('es','escuela', n=5)
    #['escuelala', 'academia', 'universidad', 'laescuela']

    '''
    
    # download and load the fasttext model if it's not already loaded
    try:
        fasttext_models[lang]
    except KeyError:
        fasttext_models[lang] = load_model(lang)
    
    # augments_fasttext defaults to using the annoy library to find nearest neighbors,
    # if annoy==False is passed into the function, then the fasttext library will be used.
    if annoy:
        # create AnnoyIndex if not already created
        try:
            annoy_indices[lang]
        except KeyError:
            annoy_indices[lang] = AnnoyIndex(300, 'angular')

        # if annoy index has not been saved for this language yet,
        # populate annoy index with vectors from corresponding fasttext model
        try:
            annoy_indices[lang].load('{0}.ann'.format(lang))
        # OSError occurs if index is not already saved to disk
        except OSError:
            load_index(lang, fasttext_models, annoy_indices)

        # find the most similar words using annoy library
        n_nearest_neighbor_indices = annoy_indices[lang].get_nns_by_vector(fasttext_models[lang][word], n)
        n_nearest_neighbors = []
        for i in range(n):
            n_nearest_neighbors.append(fasttext_models[lang].words[n_nearest_neighbor_indices[i]])
        words = ' '.join([ word for word in n_nearest_neighbors ])
    else:
        # find the most similar words using fasttext library
        topn = fasttext_models[lang].get_nearest_neighbors(word, k=n)
        words = ' '.join([ word for (rank, word) in topn ])



   # # create AnnoyIndex 
   # index = AnnoyIndex(300, 'angular')

   # # if annoy index has not been created for this language yet,
   # # populate with vectors from corresponding fasttext model 
   # try:
   #     index.load('{0}.ann'.format(lang))
   # except:
   #     i = 0
   #     for j in fasttext_models[lang].words:
   #         v = fasttext_models[lang][j]
   #         index.add_item(i,v)
   #         i += 1
   #     index.build(10)
   #     index.save('{0}.ann'.format(lang))
   #     index.load('{0}.ann'.format(lang))

   # #find the most similar words using annoy index
   # try:
   #     n_nearest_neighbor_indices = index.get_nns_by_vector(fasttext_models[lang][word], n)
   #     n_nearest_neighbors = []
   #     for i in range(n):
   #         n_nearest_neighbors.append(fasttext_models[lang].words[n_nearest_neighbor_indices[i]])
   #     words = ' '.join([ word for word in n_nearest_neighbors ])
   #     print("words = ", words)
   # except KeyError:
   #     return []

    #find the most similar words
   # try:
   #     topn = fasttext_models[lang].get_nearest_neighbors(word, k=n)
   #     words = ' '.join([ word for (rank, word) in topn ])
   # except KeyError:
   #     return []

    # lemmatize the results so that they'll be in the search document's vocabulary
    words = lemmatize(lang, words, add_positions=False, config=config).split()
    words = list(filter(lambda w: len(w)>1 and w != word, words))[:n]


    return words

# The following imports as well as suppress_stdout_stderr() are taken from stackoverflow:
# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
# In this project, suppress_stdout_stderr() is used for redirecting stdout and stderr when downloading gensim and fasttext models.
# Without this, the doctests fail due to stdout and stderr from gensim and fasttext model downloads.
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """Context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

print("tsquery  single quotes baby boy = ", to_tsquery('en', 'baby boy', augment_with=augments_fasttext))
print("tsquery double quotes baby boy = ", to_tsquery('en', '"baby boy"', augment_with=augments_fasttext))
print("tsquery baby boy (school | home) !weapon = ", to_tsquery('en', '"baby boy" (school | home) !weapon', augment_with=augments_fasttext))
