from datasets import metadict_factory, imagedict_factory


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_code", type=str, default="toys")
    args = parser.parse_args()
    metadict = metadict_factory(args)
    imagedict = imagedict_factory(args)

    for k, v in islice(metadict.items(), 10):
        print(k, v)
    for k, v in islice(imagedict.items(), 10):
        print(k, v)

