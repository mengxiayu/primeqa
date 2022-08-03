
import os
from stanza.server import CoreNLPClient
os.environ['CORENLP_HOME'] = '/dccstor/myu/.stanfordnlp_resources/stanford-corenlp-4.4.0'

with CoreNLPClient(annotators=['openie'], 
                   memory='4G', endpoint='http://localhost:7090', be_quiet=True) as client:
    text = "Albert Einstein was a German-born theoretical physicist."
    document = client.annotate(text)
    for sent in document.sentence:
        for triple in sent.openieTriple:
            print(triple.subject, triple.relation, triple.object)

print("\nThe server should be stopped upon exit from the \"with\" statement.")