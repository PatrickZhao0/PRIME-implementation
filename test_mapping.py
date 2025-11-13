from datasets import metadict_factory, imagedict_factory
from config import *

if __name__ == "__main__":
    set_template(args)
    metadict = metadict_factory(args)
    imagedict = imagedict_factory(args)

    print (metadict[1])
    print (imagedict[1])
