import pandas as pd
from os.path import join
import numpy as np
import torch.utils.data as data
import sys
import torch
import os
from sklearn.neighbors import NearestNeighbors

from datasets.datahelpers import default_loader, imresize

default_cities = {
    "train": [
        "zurich",
        "london",
        "boston",
        "melbourne",
        "amsterdam",
        "helsinki",
        "tokyo",
        "toronto",
        "saopaulo",
        "moscow",
        "trondheim",
        "paris",
        "bangkok",
        "budapest",
        "austin",
        "berlin",
        "ottawa",
        "phoenix",
        "goa",
        "amman",
        "nairobi",
        "manila",
    ],
    "val": ["cph", "sf"],
    "test": ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"],
}

default_cities_debug = {
    "train": ["zurich", "london"],
    "val": ["cph", "sf"],
    "test": ["miami"],
}


class BaseDataset(data.Dataset):
    def __init__(
        self,
        name,
        mode="train",
        imsize=None,
        transform=None,
        loader=default_loader,
        posDistThr=10,
        negDistThr=25,
        root_dir="data",
        cities="",
    ):

        root_dir = os.path.join(root_dir, "MSLS")

        if cities == "debug":
            self.cities = default_cities_debug[mode]
        elif cities in default_cities:
            self.cities = default_cities[cities]
        elif cities == "":
            self.cities = default_cities[mode]
        else:
            self.cities = cities.split(",")

        self.dbImages = []
        self.qImages = []
        self.qidxs = []
        self.pidxs = []
        self.clusters = []

        # hyper-parameters
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.imsize = imsize

        self.transform = transform
        self.loader = loader

        # flags
        self.name = name
        self.exclude_panos = True
        self.mode = mode

        # other
        self.transform = transform

        # load data
        for city in self.cities:
            print("=====> {}".format(city))

            subdir = "test" if city in default_cities["test"] else "train_val"

            # get len of images from cities so far for indexing
            _lenQ = len(self.qImages)
            _lenDb = len(self.dbImages)

            # when GPS / UTM is available
            if self.mode in ["train", "val", "test"]:

                # load query / database data
                qData = pd.read_csv(
                    join(root_dir, subdir, city, "query", "postprocessed.csv"),
                    index_col=0,
                )
                dbData = pd.read_csv(
                    join(root_dir, subdir, city, "database", "postprocessed.csv"),
                    index_col=0,
                )

                # filter based on panorama data
                if self.exclude_panos:

                    # load query / database data
                    qDataRaw = pd.read_csv(
                        join(root_dir, subdir, city, "query", "raw.csv"), index_col=0
                    )
                    dbDataRaw = pd.read_csv(
                        join(root_dir, subdir, city, "database", "raw.csv"), index_col=0
                    )

                    qData = qData.loc[(qDataRaw["pano"] == False).values, :]
                    dbData = dbData.loc[(dbDataRaw["pano"] == False).values, :]

                # append image keys with full path
                self.qImages.extend(
                    [
                        join(root_dir, subdir, city, "query", "images", key + ".jpg")
                        for key in qData["key"].values
                    ]
                )
                self.dbImages.extend(
                    [
                        join(root_dir, subdir, city, "database", "images", key + ".jpg")
                        for key in dbData["key"].values
                    ]
                )

                # utm coordinates
                utmQ = qData[["easting", "northing"]].values.reshape(-1, 2)
                utmDb = dbData[["easting", "northing"]].values.reshape(-1, 2)

                # find positive images for training
                neigh = NearestNeighbors(algorithm="brute")
                neigh.fit(utmDb)
                _, pI = neigh.radius_neighbors(utmQ, self.posDistThr)

                if mode == "train":
                    _, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

                for qidx in range(len(qData)):

                    # the query image has at least one positive
                    if len(pI[qidx]) > 0:

                        self.qidxs.append(qidx + _lenQ)
                        self.pidxs.append([p + _lenDb for p in pI[qidx]])

                        # in training we have two thresholds, one for finding positives and one for finding images that we are certain are negatives.
                        if self.mode == "train":

                            self.clusters.append([n + _lenDb for n in nI[qidx]])

                self.utmQ = (
                    np.concatenate([self.utmQ, utmQ], axis=0)
                    if hasattr(self, "utmQ")
                    else utmQ
                )
                self.utmDb = (
                    np.concatenate([self.utmDb, utmDb], axis=0)
                    if hasattr(self, "utmDb")
                    else utmDb
                )

            # when GPS / UTM / pano info is not available
            elif self.mode in ["test2"]:
                raise NotImplementedError

            # if a combination of cities, task and subtask is chosen, where there are no query/database images, then exit
            if len(self.dbImages) == 0:
                print("Exiting...")
                print(
                    "A combination of cities, task and subtask have been chosen, where there are no query/database images."
                )
                print("Try choosing a different subtask or more cities")
                sys.exit()

        # cast to np.arrays for indexing during training
        self.dbImages = np.asarray(self.dbImages)
        self.qImages = np.asarray(self.qImages)
        self.qidxs = np.asarray(self.qidxs)
        self.pidxs = np.asarray(self.pidxs, dtype=object)
        self.clusters = np.asarray(self.clusters, dtype=object)

        # else:
        #    raise(RuntimeError("Unknown dataset name!"))

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):

        raise NotImplementedError

    def __repr__(self):
        fmt_str = self.__class__.__name__ + "\n"
        fmt_str += "    Name and mode: {} {}\n".format(self.name, self.mode)
        fmt_str += "    Number of images: {}\n".format(len(self.dbImages))
        fmt_str += "    Number of training tuples: {}\n".format(len(self.qpool))
        fmt_str += "    Number of negatives per tuple: {}\n".format(self.nnum)
        fmt_str += "    Number of tuples processed in an epoch: {}\n".format(self.qsize)
        fmt_str += "    Pool size for negative remining: {}\n".format(self.poolsize)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class TrainDataset(BaseDataset):
    def __init__(
        self,
        name,
        mode,
        imsize=None,
        transform=None,
        loader=default_loader,
        posDistThr=10,
        negDistThr=25,
        root_dir="data",
        cities="",
    ):
        super().__init__(
            name,
            mode,
            imsize,
            transform,
            loader,
            posDistThr,
            negDistThr,
            root_dir,
            cities,
        )

    def __len__(self):

        return len(self.qidxs)

    def __getitem__(self, index):

        qidx = self.qidxs[index]
        pidx = self.pidxs[index]

        pidx = np.random.choice(pidx, 1)[0]

        qpath, utmQ = self.qImages[qidx], self.utmQ[qidx]
        ppath, utmDb = self.dbImages[pidx], self.utmDb[pidx]

        output = []
        output.append(self.loader(qpath))
        output.append(self.loader(ppath))

        target = []
        target.append(utmQ)
        target.append(utmDb)

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]

        if self.transform is not None:
            output = [self.transform(output[i]) for i in range(len(output))]

        return output, target


class TestDataset(BaseDataset):
    def __init__(
        self,
        name,
        mode,
        imsize=None,
        transform=None,
        loader=default_loader,
        posDistThr=10,
        negDistThr=25,
        root_dir="data",
        cities="",
    ):
        super().__init__(
            name,
            mode,
            imsize,
            transform,
            loader,
            posDistThr,
            negDistThr,
            root_dir,
            cities,
        )

    def __len__(self):
        # the dataset is the queries followed by the database images
        return len(self.qidxs) + len(self.dbImages)

    def __getitem__(self, index):

        if index < len(self.qidxs):
            path = self.qImages[self.qidxs[index]]
            utm = self.utmQ[self.qidxs[index]]
            index = [index, -1]
        else:
            path = self.dbImages[index - len(self.qidxs)]
            utm = self.utmDb[index - len(self.qidxs)]
            index = [-1, index - len(self.qidxs)]

        img = self.loader(path)

        if self.imsize is not None:
            img = imresize(img, self.imsize)

        if self.transform is not None:
            img = self.transform(img)

        return img, index, utm
