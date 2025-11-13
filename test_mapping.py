from datasets import metadict_factory, imagedict_factory
from config import *

if __name__ == "__main__":
    set_template(args)
    metadict = metadict_factory(args)
    imagedict = imagedict_factory(args)

    for k, v in islice(metadict.items(), 10):
        print(k, v)
    for k, v in islice(imagedict.items(), 10):
        print(k, v)

